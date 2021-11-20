import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable
from torch.optim import lr_scheduler


class Lscae(nn.Module):
    def __init__(self, input_dim: int = None, device=None, **kwargs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.cfg = OmegaConf.create({
            "input_dim": None,          # Dimension of input dataset (total #features)
            "k_selected": 2,            # Number of selected features
            "decoder_lr": 1e-3,         # Decoder learning rate
            "selector_lr": 1e-1,        # Concrete layer learning rate
            "min_lr": 1e-5,             # Minimal layer learning rate
            "weight_decay": 0,          # l2 weight penalty
            "batch_size": 64,           # Minibatch size
            "hidden_dim": 128,          # Hidden layers size
            "model": 'lscae',           # lscae | cae | ls
            "scale_k": 2,               # Number of neighbors for computation of local scales for the kernel
            "laplacian_k": 50,          # Number of neighbors of each pooint, used for computation of the Laplacian
            "start_temp": 10,           # Initial temperature
            "min_temp": 1e-2,           # Final temperature
            "rec_lambda": .5,           # Balance between reconstruction and LS terms
            "fr_penalty": 0,            # Feature redundancy penalty
            "num_epochs": 500,          # Number of training epochs
            "verbose": True             # Whether to print to console during training
        })
        self.cfg.input_dim = input_dim
        self.cfg.update((key, kwargs['kwargs'][key]) for key in self.cfg.keys() if key in kwargs['kwargs'].keys())
        assert self.cfg.laplacian_k >= self.cfg.scale_k, 'laplacian_k needs to be greater than of equal to scale_k'
        assert self.cfg.laplacian_k <= self.cfg.batch_size, 'laplacian_k needs to be less than or equal to than of equal to the batch size'
        assert self.cfg.input_dim is not None, 'Provide input_dim argument to Lscae.__init__'

        self.selector = SelectLayer(self.cfg).to(self.device)
        self.decoder = Decoder(self.cfg).to(self.device)
        self.model = self.cfg.model
        self.k_selected = self.cfg.k_selected

        self.optim = torch.optim.Adam([{'params': self.decoder.parameters(), 'lr': self.cfg.decoder_lr},
                                       {'params': self.selector.parameters(), 'lr': self.cfg.selector_lr}],
                                      lr=self.cfg.decoder_lr,
                                      betas=(0.5, 0.999),
                                      weight_decay=self.cfg.weight_decay)

        self.scheduler = lr_scheduler.LambdaLR(self.optim, lr_lambda=self.lambda_rule)

    def forward(self, x, epoch=None):
        selected_feats, weights = self.selector(x, epoch)
        recon = self.decoder(selected_feats)
        return weights, recon

    def get_selected_feats(self):
        return self.selector.get_selected_feats().detach().cpu().numpy()

    def get_selection_probs(self, epoch):
        return self.selector.get_weights(epoch=epoch).detach().cpu().numpy()

    @staticmethod
    def lambda_rule(i) -> float:
        """ stepwise learning rate calculator """
        lr_decay_factor = .1
        decay_step_size = 100
        exponent = int(np.floor((i + 1) / decay_step_size))
        return np.power(lr_decay_factor, exponent)

    @staticmethod
    def compute_diff_laplacian(W):
        """
        Computes random walk Laplacian matrix
        W: kernel tensor
        """
        row_sums = torch.sum(W, dim=1)
        D = torch.diag(row_sums)
        L = torch.matmul(torch.inverse(D), W)
        return L

    @staticmethod
    def compute_kernel_mat(D, scale, Ids=None, device=torch.device('cpu')):
        """
        Computes RBF kernal matrix
        args:
           D: nxn tenosr of squared distances
           scale: standard dev
           Ids: output of nnsearch
        """

        if isinstance(scale, float):
            # global scale
            W = torch.exp(-torch.pow(D, 2) / (scale ** 2))
        else:
            # local scales
            W = torch.exp(-torch.pow(D, 2) / (torch.tensor(scale, device=device).float().clamp_min(1e-7) ** 2))
        if Ids is not None:
            n, k = Ids.shape
            mask = torch.zeros([n, n], device=device)
            for i in range(len(Ids)):
                mask[i, Ids[i]] = 1
            W = W * mask
        sym_W = (W + torch.t(W)) / 2.
        return sym_W

    @staticmethod
    def compute_scale(D, k=2, med=False):
        """
        Computes scale as the max distance to the k neighbor
        args:
            Dis: nxk' numpy array of distances (output of nn_search)
            k: number of neighbors
        """
        if not med:
            scale = np.max(D[:, k - 1])
        else:
            scale = np.median(D[:, k - 1])
        return scale

    @staticmethod
    def compute_dist_mat(X, Y=None, device=torch.device("cpu")):
        """
        Computes nxm matrix of squared distances
        args:
            X: nxd tensor of data points
            Y: mxd tensor of data points (optional)
        """
        if Y is None:
            Y = X

        X = X.to(device=device)
        Y = Y.to(device=device)
        dtype = X.data.type()
        dist_mat = Variable(torch.Tensor(X.size()[0], Y.size()[0]).type(dtype)).to(device=device)

        for i, row in enumerate(X.split(1)):
            r_v = row.expand_as(Y)
            sq_dist = torch.sum((r_v - Y) ** 2, 1)
            dist_mat[i] = sq_dist.view(1, -1)
        return dist_mat

    @staticmethod
    def nn_search(X, Y=None, k=10):
        """
        Computes nearest neighbors in Y for points in X
        args:
            X: nxd tensor of query points
            Y: mxd tensor of data points (optional)
            k: number of neighbors
        """
        if Y is None:
            Y = X
        X = X.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(Y)
        Dis, Ids = nbrs.kneighbors(X)
        return Dis, Ids

    def train_step(self, x, current_epoch):
        """
        training procedure for LS-CAE
        args:
            x: nxd tensor (minibatch)
            current_epoch: current training epoch
        """

        selection_probs, recon = self(x, current_epoch)
        xx = x * selection_probs.max(dim=1)[0]

        # compute Laplacian
        D = self.compute_dist_mat(xx, device=self.device)
        Dis, Ids = self.nn_search(xx, k=self.cfg.laplacian_k)
        scale = self.compute_scale(Dis, k=self.cfg.scale_k)
        W = self.compute_kernel_mat(D, scale, Ids=None, device=self.device)
        L = self.compute_diff_laplacian(W)
        L2 = torch.matmul(L, L)

        # Laplacian score term
        FLF = torch.matmul(torch.matmul(torch.t(xx), L2), xx)
        ls_loss = -torch.trace(FLF) / (self.cfg.batch_size * self.k_selected)

        # recon loss
        rec_loss = torch.nn.MSELoss()(x, recon)

        # loss
        if self.model == 'cae':
            loss = rec_loss
        elif self.model == 'ls':
            loss = ls_loss
        elif self.model == 'lscae':
            loss = rec_loss / rec_loss.item() + ls_loss / np.abs(ls_loss.item())

        # feature redundancy penalty
        max_occurances = torch.max(torch.sum(self.selector.get_weights(epoch=current_epoch), dim=1))
        fr_penalty = torch.max(torch.tensor(0).to(device=self.device), max_occurances - 1) * self.cfg.fr_penalty
        loss += fr_penalty

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), ls_loss.item(), rec_loss.item()

    def select_features(self, dataloader: torch.utils.data.DataLoader):

        for epoch in range(self.cfg.num_epochs):
            batch_losses = []
            ls_losses = []
            recon_losses = []
            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(device=self.device)
                batch_loss, ls_loss, recon_loss = self.train_step(batch_x, epoch)
                batch_losses.append(batch_loss)
                ls_losses.append(ls_loss)
                recon_losses.append(recon_loss)
            epoch_loss = np.mean(batch_losses)
            epoch_ls_loss = np.mean(ls_losses)
            epoch_recon_loss = np.mean(recon_losses)

            if epoch % 5 == 0:
                print(f'Epoch {epoch + 1}\{self.cfg.num_epochs}, loss: {epoch_loss:.3f}, ls loss: {epoch_ls_loss:.5f}, recon loss: {epoch_recon_loss:.3f}')

            if epoch % 20 == 0 and self.cfg.verbose:
                print('Selection probs: \n ', self.selector.get_weights(epoch).max(dim=1)[0].detach().cpu().numpy(), '\n')

            self.update_lr()
        print('Finished training LS-CAE')
        selected_features = set(self.get_selected_feats())
        print('Selected features:', selected_features)
        return selected_features

    def update_lr(self):
        """ Learning rate updater """
        self.scheduler.step()
        lr = self.optim.param_groups[0]['lr']
        if lr < self.cfg.min_lr:
            self.optim.param_groups[0]['lr'] = self.cfg.min_lr
            lr = self.optim.param_groups[0]['lr']
        print(f'LS-CAE learning rate = {lr:.7f}')


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_features = cfg.k_selected
        self.output_features = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, self.hidden_dim, bias=False),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.LeakyReLU(.2, True),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.LeakyReLU(.2, True),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.output_features, bias=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SelectLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_features = self.cfg.input_dim
        self.output_features = self.cfg.k_selected
        self.num_epochs = self.cfg.num_epochs
        self.start_temp = self.cfg.start_temp
        self.min_temp = torch.tensor(self.cfg.min_temp)
        self.logits = torch.nn.Parameter(torch.zeros(self.input_features, self.output_features), requires_grad=True)

    def current_temp(self, epoch, sched_type='exponential'):
        schedules = {
            'exponential': torch.max(self.min_temp, self.start_temp * ((self.min_temp / self.start_temp) ** (epoch / self.num_epochs))),
            'linear': torch.max(self.min_temp, self.start_temp - (self.start_temp - self.min_temp) * (epoch / self.num_epochs)),
            'cosine': self.min_temp + 0.5 * (self.start_temp - self.min_temp) * (1. + np.cos(epoch * math.pi / self.num_epochs))
        }
        return schedules[sched_type]

    def forward(self, x, epoch=None):
        from torch.distributions.uniform import Uniform
        uniform_pdfs = Uniform(low=1e-6, high=1.).sample(self.logits.size()).to(x.device)
        gumbel = -torch.log(-torch.log(uniform_pdfs))

        if self.training:
            temp = self.current_temp(epoch)
            noisy_logits = (self.logits + gumbel) / temp
            weights = F.softmax(noisy_logits / temp, dim=0)
            x = x @ weights
        else:
            weights = F.one_hot(torch.argmax(self.logits, dim=0), self.input_features).float()
            x = x @ weights.T
        return x, weights

    def get_weights(self, epoch):
        temp = self.current_temp(epoch)
        return F.softmax(self.logits / temp, dim=0)

    def get_selected_feats(self):
        feats = torch.argmax(self.logits, dim=0)
        return feats

