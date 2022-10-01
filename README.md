# ResGrad: Residual Denoising Diffusion Probabilistic Models for Text to Speech

## Usages:
```
import torch
from model import GradLogPEstimator2d
from diffusion import GaussianDiffusion


dim = 64
pe_scale = 1000
spk_emb_dim = 64
n_spks = 0

estimator = GradLogPEstimator2d(dim, n_spks=n_spks,
                                             spk_emb_dim=spk_emb_dim,
                                             pe_scale=pe_scale)
  
diff = GaussianDiffusion(estimator)

```
