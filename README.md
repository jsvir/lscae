# Laplacian Score-regularized Concrete Autoencoders

## Requirements:

* torch >= 1.9
* scikit-learn >= 0.24
* omegaconf >= 2.0.6
* scipy >= 1.6.0
* matplotlib

## How to use:

Please see an example [here](https://github.com/jsvir/lscae/blob/master/example.ipynb)

`import lscae`
`import torch`
`# define your dataset`
`# define you cfg parameters`
`dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)`
`lscae.Lscae(kwargs=cfg).select_features(dataloader)`
