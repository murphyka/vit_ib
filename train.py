import argparse
import math
import os
import time

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.cuda.init()
time.sleep(2)

import matplotlib.pyplot as plt
import random
import numpy as np

import models_final as models

import timm
from timm.models.vision_transformer import VisionTransformer, Attention
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

print("Device:", torch.cuda.get_device_name(0))
print("Current device:", torch.cuda.current_device())

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

@dataclass
class Args:
    data: str
    model: str = "vit_tiny_patch16_224"
    img_size: int = 224
    patch_size: int = 16
    epochs: int = 50
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs_lr: int = 10
    warmup_epochs_beta: int = 10
    beta_scheduler_type: str = "linear"
    beta: float = 1e-4
    workers: int = 8
    seed: int = 42
    amp: bool = True
    output: str = "./out_IN100"
    save_interval: int = 1
    ib_mlp_hidden_dim: Optional[int] = None
    attn_head_bottleneck_dim: Optional[int] = None
    no_ib_attn: bool = False
    sqrt_kl: bool = False


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Train ViT with token-IB")
    p.add_argument("--data", type=str, help="Path to dataset root (ImageFolder)", default="/workspace/DATA/IN100")
    p.add_argument("--model", type=str, default="vit_tiny_patch16_224", help="timm ViT model name")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs-lr", type=int, default=10)
    p.add_argument("--warmup-epochs-beta", type=int, default=10)
    p.add_argument("--beta-scheduler-type", type=str, default="linear", choices=["linear", "log"], help="Type of beta scheduler")
    p.add_argument("--beta", type=float, default=1e-4, help="Weight for IB KL term")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision AMP")
    p.add_argument("--output", type=str, default="./out_IN100")
    p.add_argument("--save-interval", type=int, default=10, help="Save visualizations, checkpoints every N epochs")
    p.add_argument("--ib-mlp-hidden-dim", type=int, default=None, help="Hidden dimension for IB MLP")
    p.add_argument("--attn-head-bottleneck-dim", type=int, default=None, help="Hidden dimension for IB MLP")
    p.add_argument("--no-ib-attn", action="store_true", help="Turn off the attention head IB")
    p.add_argument("--sqrt-kl", action="store_true", help="Sqrt kl to encourage consolidation")
    args = p.parse_args()
    return Args(
        data=args.data,
        model=args.model,
        img_size=args.img_size,
        patch_size=args.patch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs_lr=args.warmup_epochs_lr,
        warmup_epochs_beta=args.warmup_epochs_beta,
        beta_scheduler_type=args.beta_scheduler_type,
        beta=args.beta,
        workers=args.workers,
        seed=args.seed,
        amp=(not args.no_amp),
        output=args.output,
        save_interval=args.save_interval,
        ib_mlp_hidden_dim=args.ib_mlp_hidden_dim,
        attn_head_bottleneck_dim=args.attn_head_bottleneck_dim,
        no_ib_attn=args.no_ib_attn,
        sqrt_kl=args.sqrt_kl,

    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(root: str, img_size: int, batch_size: int, workers: int):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 256/224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # Expect ImageFolder layout: root/train/<class> and root/val/<class>
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transforms)
    train_eval_ds = datasets.ImageFolder(train_dir, transform=eval_transforms)

    ## Fix the seed so that the val images are the same each time we visualize (but shuffled so that class variation is present)
    g = torch.Generator()
    g.manual_seed(42) 
    perm = torch.randperm(len(val_ds), generator=g).tolist()  # one-time shuffle
    val_subset  = torch.utils.data.Subset(val_ds, perm)
    val_sampler = torch.utils.data.SequentialSampler(val_subset)  # fixed order thereafter

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, 
    pin_memory=True, drop_last=True, persistent_workers=(workers>0), prefetch_factor=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler, num_workers=workers, 
    pin_memory=True, drop_last=False, persistent_workers=(workers>0), prefetch_factor=4)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, num_workers=workers, 
    pin_memory=True, drop_last=False, persistent_workers=(workers>0), prefetch_factor=4)
    return train_loader, val_loader, train_eval_loader, len(train_ds.classes)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    """ Cosine LR scheduler with linear warmup. Returns a list with per-iteration LR. """
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep
    lr_schedule = []
    for it in range(total_iters):
        if it < warmup_iters and warmup_iters > 0:
            lr = base_value * it / warmup_iters
        else:
            t = (it - warmup_iters) / max(1, total_iters - warmup_iters)
            lr = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * t))
        lr_schedule.append(lr)
    return lr_schedule


def linear_ramp_scheduler(base_value, warmup_epochs, total_epochs, niter_per_ep):
    """ Linear warmup scheduler. Returns a list with per-iteration beta. """
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = total_epochs * niter_per_ep
    beta_schedule = []
    for it in range(total_iters):
        if it < warmup_iters and warmup_iters > 0:
            beta = base_value * it / warmup_iters
        else:
            beta = base_value
        beta_schedule.append(beta)
    return beta_schedule
    
def log_ramp_scheduler(base_value, zero_value, warmup_epochs, total_epochs, niter_per_ep):
    """ Log warmup scheduler. Returns a list with per-iteration beta. """
    assert zero_value > 1e-16, "Zero value must be greater than 1e-16"
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = total_epochs * niter_per_ep
    beta_schedule = []
    for it in range(total_iters):
        if it < warmup_iters and warmup_iters > 0:
            beta = np.exp(np.log(zero_value) + (np.log(base_value)-np.log(zero_value)) * it / warmup_iters)
        else:
            beta = base_value
        beta_schedule.append(beta)
    return beta_schedule


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_param_groups(model, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith(".bias")
            or param.ndim == 1
            or ".ib." in name
        ):
            no_decay.append(param)
            
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
        
def safe_model_to_device(model, device, max_retries=3, wait=5):
    for attempt in range(max_retries):
        try:
            return model.to(device)
        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < max_retries - 1:
                print(f"CUDA error on attempt {attempt+1}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

def main():
    args = parse_args()
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | AMP: {args.amp}")
    from pprint import pprint
    print("Arguments:")
    pprint(vars(args), sort_dicts=False)

    # Data
    train_loader, val_loader, train_eval_loader, num_classes = build_dataloaders(args.data, args.img_size, args.batch_size, args.workers)
    print(f"Classes: {num_classes} | Train iters/epoch: {len(train_loader)} | Val iters: {len(val_loader)}")

    # Base ViT
    if args.patch_size != 16:
        print(f"Altering default patch size to {args.patch_size} for model {args.model}")
    if args.img_size != 224:
        print(f"Altering default image size to {args.img_size} for model {args.model}")

    base = timm.create_model(args.model, pretrained=False, num_classes=num_classes, img_size=args.img_size, patch_size=args.patch_size)
    if not isinstance(base, VisionTransformer):
        raise ValueError(f"Model {args.model} is not a VisionTransformer. Try a timm ViT like vit_tiny_patch16_224.")

    # Build per-head IBs inside attention
    if args.no_ib_attn:
        for i, blk in enumerate(base.blocks):
            blk.attn = models.AttentionWithHeadIB(blk.attn)
        wrapped_blocks = nn.ModuleList([models.BlockWithAttnIB(b) for b in base.blocks])
    else:
        for i, blk in enumerate(base.blocks):
            ib_head = models.HeadwiseVIBED(head_dim=blk.attn.head_dim, bottleneck_dim=args.attn_head_bottleneck_dim, num_heads=blk.attn.num_heads,
                                hidden_enc=args.ib_mlp_hidden_dim,
                                hidden_dec=args.ib_mlp_hidden_dim,
                                init_logvar=-6.0, clamp=(-10, 10), small_init_dec=False)
            blk.attn = models.AttentionWithHeadIB(blk.attn, ib=ib_head)
    
        # Wrap blocks to accumulate KL
        wrapped_blocks = nn.ModuleList([models.BlockWithAttnIB(b) for b in base.blocks])

   
    model = safe_model_to_device(models.ViTWithHeadIB(base, wrapped_blocks), device)

    param_groups = get_param_groups(model, args.weight_decay)
    

    optim = torch.optim.AdamW(param_groups, lr=args.lr)
    lr_sched = cosine_scheduler(args.lr, final_value=1e-6, epochs=args.epochs, niter_per_ep=len(train_loader), warmup_epochs=args.warmup_epochs_lr)

    if args.warmup_epochs_beta > 0:
        if args.beta_scheduler_type == "linear":
            beta_sched = linear_ramp_scheduler(args.beta, args.warmup_epochs_beta, args.epochs, niter_per_ep=len(train_loader))
        elif args.beta_scheduler_type == "log":
            beta_ramp_start = 1e-8
            beta_sched = log_ramp_scheduler(args.beta, beta_ramp_start, args.warmup_epochs_beta, args.epochs, niter_per_ep=len(train_loader))
        else:
            raise ValueError(f"Invalid beta scheduler type: {args.beta_scheduler_type}")
    else:
        beta_sched = [args.beta] * args.epochs * len(train_loader)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    # scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    criterion = F.cross_entropy

    global_iter = 0
    train_loss_series = []
    train_ce_series = []
    train_kl_series = []
    train_top1_series = []
    train_top5_series = []
    val_loss_series = []
    val_top1_series = []
    val_top5_series = []
    val_kl_series = []
    SAFETY_EPSILON = 1e-8
    for epoch in range(args.epochs):
        ct = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_kl = 0.0

        optim.zero_grad(set_to_none=True)

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            for param_group in optim.param_groups:
                param_group['lr'] = lr_sched[global_iter]

            beta = beta_sched[global_iter]
            global_iter += 1

            with torch.amp.autocast('cuda', enabled=args.amp):
                logits, kl_logs = model(images)  # kl: [B] or None
                ce_loss = criterion(logits, targets)
                kl_loss = torch.concat(kl_logs, dim=1)
                if args.sqrt_kl:
                    kl_loss = torch.sqrt(torch.clamp(kl_loss, min=0.0).mean(-1) + SAFETY_EPSILON).mean()  ## sum over patches linearly so that part is unchanged, but then sum over the sqrt on heads so they consolidate (and average over batch)
                else:
                    kl_loss = kl_loss.mean()
                loss = ce_loss + beta * kl_loss
                
            scaler.scale(loss).backward()
            
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_kl += kl_loss.item()


        print(f'Epoch {epoch+1}/{args.epochs}. Time taken: {time.time()-ct:.3f} sec.  loss: {epoch_loss/(i+1):.3f}, ce: {epoch_ce/(i+1):.3f}, kl: {epoch_kl/(i+1):.3f}')
        train_loss_series.append(epoch_loss/(i+1))
        train_ce_series.append(epoch_ce/(i+1))
        train_kl_series.append(epoch_kl/(i+1))

        # Validation
        model.eval()
        val_top1 = 0.0
        val_top5 = 0.0
        val_loss = 0.0
        val_kl = 0.0
        train_top1_eval = 0.0
        train_top5_eval = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=args.amp):
                    logits, kl_logs = model(images)
                    ce_loss = F.cross_entropy(logits, targets)
                    kl_loss = torch.concat(kl_logs, dim=1)
                    if args.sqrt_kl:
                        kl_loss = torch.sqrt(torch.clamp(kl_loss, min=0.0).mean(-1) + SAFETY_EPSILON).mean()  ## sum over patches linearly so that part is unchanged, but then sum over the sqrt on heads so they consolidate (and average over batch)
                    else:
                        kl_loss = kl_loss.mean()
                    loss = ce_loss + beta * kl_loss
                top1, top5 = accuracy(logits, targets, topk=(1, 5))
                val_top1 += top1.item()
                val_top5 += top5.item()
                val_loss += loss.item()
                val_kl += kl_loss.item()
            if epoch % 10 == 0:
                for images, targets in train_eval_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=args.amp):
                        logits, kl_logs = model(images)
                        ce_loss = F.cross_entropy(logits, targets)
                        kl_loss = torch.concat(kl_logs, dim=1)
                        if args.sqrt_kl:
                            kl_loss = torch.sqrt(torch.clamp(kl_loss, min=0.0).mean(-1) + SAFETY_EPSILON).mean()  ## sum over patches linearly so that part is unchanged, but then sum over the sqrt on heads so they consolidate (and average over batch)
                        else:
                            kl_loss = kl_loss.mean()
                        loss = ce_loss + beta * kl_loss
                    top1, top5 = accuracy(logits, targets, topk=(1, 5))
                    train_top1_eval += top1.item()
                    train_top5_eval += top5.item()
        val_top1 /= len(val_loader)
        val_top5 /= len(val_loader)
        val_loss /= len(val_loader)
        val_loss_series.append(val_loss)
        val_top1_series.append(val_top1)
        val_top5_series.append(val_top5)
        val_kl_series.append(val_kl/len(val_loader))
        
        train_top1_eval /= len(train_eval_loader)
        train_top5_eval /= len(train_eval_loader)
        train_top1_series.append(train_top1_eval)
        train_top5_series.append(train_top5_eval)
        print(f"[Epoch {epoch+1}] TrainTop1={train_top1_eval:.2f}  TrainTop5={train_top5_eval:.2f}  ValTop1={val_top1:.2f}  ValTop5={val_top5:.2f}  ValLoss={val_loss:.3f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            if args.save_checkpoints:
                ckpt = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                }
                ckpt_path = os.path.join(output_dir, f"model_beta{args.beta}_epochs{args.epochs}_ckpt_epoch_{epoch+1}.pth")
                torch.save(ckpt, ckpt_path)

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_beta{args.beta}_epochs{args.epochs}.pth"))
    # Save the args in a file with the same name as the model
    with open(os.path.join(output_dir, f"args_beta{args.beta}_epochs{args.epochs}.txt"), "w") as f:
        for key, value in args.__dict__.items():
            f.write(f"{key}: {value}\n")

    ## Save the val and train loss, accuracy, and kl
    with open(os.path.join(output_dir, f"loss_accuracy_kl_beta{args.beta:g}_epochs{args.epochs}.txt"), "w") as f:
        for i in range(len(train_loss_series)):
            f.write(f"Train Loss: {train_loss_series[i]:.6f}\t")
            f.write(f"Train CE: {train_ce_series[i]:.6f}\t")
            f.write(f"Train KL: {train_kl_series[i]:.6f}\t")
            f.write(f"Train Top1: {train_top1_series[i]:.4f}\t")
            f.write(f"Train Top5: {train_top5_series[i]:.4f}\t")
            f.write(f"Val Loss: {val_loss_series[i]:.6f}\t")
            f.write(f"Val Top1: {val_top1_series[i]:.4f}\t")
            f.write(f"Val Top5: {val_top5_series[i]:.4f}\t")
            f.write(f"Val KL: {val_kl_series[i]:.6f}\n")

if __name__ == "__main__":
    main()
