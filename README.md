# ResGrad: Residual Denoising Diffusion Probabilistic Models for Text to Speech

## Usages:
```python
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


mel_gt = torch.ones([2, 80, 100])   # Ground truth mel
mask = torch.zeros([2, 1, 100])     # Mel mask
mel_gen = torch.ones([2, 80, 100])  # Output of FS2 

x0 = mel_gt - mel_gen

loss = diff(x0, mask, mel_gen)


# Generate samples

out = diff.sample(mask, mel_gen)

mel_gen = mel_gen + out   # Final output mel send to the vocoder
```
