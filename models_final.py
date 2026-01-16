from typing import Optional, Tuple
import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, Attention

class ViTWithHeadIB(nn.Module):
    def __init__(self, base_vit: VisionTransformer, blocks: nn.ModuleList):
        super().__init__()
        self.vit = base_vit
        self.blocks = blocks

    def forward(self, x):
        x = self.vit.patch_embed(x)
        kl_logs = []
        
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed[:, 1:]  # skip the CLS token
        x = self.vit.pos_drop(x)

        for wb in self.blocks:
            x, kl_h = wb(x)                   # kl_h: [B, H, N']
            kl_logs.append(kl_h)

        x = self.vit.norm(x)

        # Average over all tokens for logits
        logits = self.vit.head(x.mean(dim=1))

        return logits, kl_logs

class HeadwiseVIBED(nn.Module):
    """
    Variational IB with an explicit encoder->latent->decoder per attention head,
    applied BEFORE W_o (per-head messages). Works on [B, H, N, d].

    - Encoder: d -> 2r   (mu, logvar)
    - Sample:  z ~ N(mu, diag(sigma^2))
    - Decoder: r -> d
    - KL computed in latent (size r), returns [B,H,N] (optionally normalized by r)

    Args:
        head_dim:        d, per-head feature size
        bottleneck_dim:  r, bottleneck size (e.g., 16 or 32 for d=64)
        num_heads:       number of heads
        hidden_enc:      None or hidden width for encoder MLP (d->h->2r)
        hidden_dec:      None or hidden width for decoder MLP (r->h->d)
        init_logvar:     initializer for logvar bias (e.g., -6.0)
        clamp:           clamp range for logvar for stability
        small_init_dec:  initialize decoder last layer near zero to start with tiny writes
    """
    def __init__(
        self,
        head_dim: int,
        bottleneck_dim: int,
        num_heads: int,
        hidden_enc: Optional[int] = None,
        hidden_dec: Optional[int] = None,
        init_logvar: float = -6.0,
        clamp: Optional[Tuple[float, float]] = (-10.0, 10.0),
        small_init_dec: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.bottleneck_dim = bottleneck_dim
        self.clamp = clamp
        
        # --- capture bottleneck representations controls / cache ---
        self.capture: bool = False          # turn on to record the next forward
        self.keep_on_device: bool = False   # False -> move cached tensors to CPU
        self.last_mu = None                 # [B, H, N, r]
        self.last_logvar = None             # [B, H, N, r]

        def make_encoder():
            if hidden_enc is None:
                mod = nn.Linear(self.head_dim, 2 * self.bottleneck_dim)
                last = mod
            else:
                mod = nn.Sequential(
                    nn.Linear(self.head_dim, hidden_enc),
                    nn.GELU(),
                    nn.Linear(hidden_enc, 2 * self.bottleneck_dim),
                )
                last = mod[-1]
            with torch.no_grad():
                if last.bias is not None:
                    # [mu | logvar] bias -> start with small variance
                    last.bias[self.bottleneck_dim:] = init_logvar
            return mod

        def make_decoder():
            if hidden_dec is None:
                mod = nn.Linear(self.bottleneck_dim, self.head_dim)
                last = mod
            else:
                mod = nn.Sequential(
                    nn.Linear(self.bottleneck_dim, hidden_dec),
                    nn.GELU(),
                    nn.Linear(hidden_dec, self.head_dim),
                )
                last = mod[-1]
            if small_init_dec:
                with torch.no_grad():
                    if hasattr(last, "weight"):
                        # nn.init.zeros_(last.weight)
                        nn.init.normal_(last.weight, mean=0.0, std=1e-4)
                    if hasattr(last, "bias") and last.bias is not None:
                        nn.init.zeros_(last.bias)
            return mod

        self.enc_list = nn.ModuleList([make_encoder() for _ in range(num_heads)])
        self.dec_list = nn.ModuleList([make_decoder() for _ in range(num_heads)])

    def _store(self, mu_list, logvar_list, B, H, N, r):
        if not self.capture:
            # clear previous to avoid stale reads
            self.last_mu = self.last_logvar = None
            return
        mu = torch.stack(mu_list, dim=1).reshape(B, H, N, r).detach()
        lv = torch.stack(logvar_list, dim=1).reshape(B, H, N, r).detach()
        if not self.keep_on_device:
            mu, lv = mu.cpu(), lv.cpu()
        self.last_mu, self.last_logvar = mu, lv

    def forward(self, y):  # y: [B, H, N, d]
        B, H, N, d = y.shape
        assert d == self.head_dim, f"Expected head_dim={self.head_dim}, got {d}"

        # Separate enc/dec per head
        y_hat_list = []
        kl_list = []
        # For optional capture of bottleneck representations
        mu_list = []
        logvar_list = []
        for head_idx in range(H):
            flat = y[:, head_idx, :, :].reshape(B * N, d)              # [BN, d]
            out = self.enc_list[head_idx](flat)                        # [BN, 2r]
            mu, logvar = out.chunk(2, dim=-1)
            if self.clamp is not None:
                lo, hi = self.clamp
                logvar = logvar.clamp(lo, hi)
            sigma = torch.exp(0.5 * logvar)
            eps   = torch.randn_like(sigma)
            z     = mu + sigma * eps
            mu32, lv32 = mu.float(), logvar.float()
            kl_h = 0.5 * (torch.exp(lv32) + mu32.pow(2) - 1.0 - lv32)  # [BN, r]
            kl_h = kl_h.sum(dim=-1).reshape(B, N)               # [B, N]
            y_hat_h = self.dec_list[head_idx](z).reshape(B, N, d)      # [B, N, d]
            y_hat_list.append(y_hat_h)
            kl_list.append(kl_h)

            # collect if capture enabled
            if self.capture:
                mu_list.append(mu.reshape(B, N, -1))
                logvar_list.append(logvar.reshape(B, N, -1))

        y_hat = torch.stack(y_hat_list, dim=1)                  # [B, H, N, d]
        kl    = torch.stack(kl_list, dim=1)                     # [B, H, N]
        # stash latents (no overhead unless capture=True)
        if self.capture:
            self._store(mu_list, logvar_list, B, H, N, self.bottleneck_dim)
        return y_hat, kl


class AttentionWithHeadIB(nn.Module):
    """
    Reimplements timm Attention forward to expose per-head messages and apply HeadwiseIB
    BEFORE projection (W_o). Returns (y, kl) so callers can accumulate KL.
    """
    def __init__(self, attn: Attention, ib: Optional[HeadwiseVIBED] = None):
        super().__init__()
        # Reuse the original submodules/params
        self.num_heads = attn.num_heads
        self.head_dim  = attn.head_dim
        self.scale     = attn.scale
        self.qkv       = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj      = attn.proj
        self.proj_drop = attn.proj_drop
        self.ib = ib

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # [B, H, N, d]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))              # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y_heads = attn @ v                            # [B, H, N, d]  (per-head messages)

        kl = None
        if self.ib is not None:
            y_heads, kl = self.ib(y_heads)            # apply per-head IB here
        else:
            kl = torch.zeros_like(y_heads[:, :, :, 0])

        y = y_heads.transpose(1, 2).reshape(B, N, C)  # concat heads
        y = self.proj(y)                              # W_o mixes heads
        y = self.proj_drop(y)
        return y, kl


class BlockWithAttnIB(nn.Module):
    def __init__(self, blk: nn.Module):
        super().__init__()
        for n in ["norm1", "attn", "norm2", "mlp"]:
            assert hasattr(blk, n), f"Block missing {n}"
        self.blk = blk
        self.has_drop_path = hasattr(blk, "drop_path") and blk.drop_path is not None

    def forward(self, x):
        y, kl_attn = self.blk.attn(self.blk.norm1(x))
        if self.has_drop_path:
            y = self.blk.drop_path(y)
        x = x + y

        y2 = self.blk.mlp(self.blk.norm2(x))
        if self.has_drop_path:
            y2 = self.blk.drop_path(y2)
        x = x + y2
        return x, kl_attn
        
class BlockWithoutAttn(nn.Module):
    def __init__(self, blk: nn.Module):
        super().__init__()
        for n in ["norm1", "attn", "norm2", "mlp"]:
            assert hasattr(blk, n), f"Block missing {n}"
        self.blk = blk
        self.has_drop_path = hasattr(blk, "drop_path") and blk.drop_path is not None

    def forward(self, x):
        y2 = self.blk.mlp(self.blk.norm2(x))
        if self.has_drop_path:
            y2 = self.blk.drop_path(y2)
        x = x + y2
        return x, torch.zeros_like(x)