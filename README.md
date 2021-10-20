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

Prepare your dataset by applying Standard Scaler on it

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
```

Then load it as `torch.utils.data.Dataset` and run feature selection
Please see an example [here](https://github.com/jsvir/lscae/blob/master/example.ipynb)

```python
import lscae
import torch
from omegaconf import OmegaConf

# define you cfg parameters
cfg = OmegaConf.create({"input_dim": 100})

# define you dataset (Torch based)

dataset = torch.utils.data.Dataset(...)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
lscae.Lscae(kwargs=cfg).select_features(dataloader)
```
