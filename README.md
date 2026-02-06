Code accompanying *From independent patches to coordinated attention: Controlling information flow in vision transformers* ([arxiv 2026](https://arxiv.org/abs/2602.04784)).

Builds off of `timm`'s ViT-tiny to insert variational information bottlenecks at every attention-mediated write to the residual stream.  
A model can be trained with the example call in `train.sh`.   

Analysis code will be added soon.
