# Laplacian Score-regularized Concrete Autoencoders

## Requirements:

* torch >= 1.9
* scikit-learn >= 0.24
* omegaconf >= 2.0.6
* scipy >= 1.6.0
* matplotlib

## How to use:

Install the package from pypi:
`pip install lscae`

```python
import lscae
import torch
from omegaconf import OmegaConf

# define you cfg parameters
cfg = OmegaConf.create({
    "input_dim": 100 })
# define you dataset (Torch based)
dataset = torch.utils.data.Dataset(...)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
lscae.Lscae(kwargs=cfg).select_features(dataloader)
```

Please see the full example [here](https://github.com/jsvir/lscae/blob/master/example.ipynb)
