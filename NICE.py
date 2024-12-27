import torch
import torch.nn as nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transforms import AffineTransform

# D is the dimension of the data, preserved through all transformations
class StandardLogisticDistribution(nn.Module):
    def __init__(self, D):
        super().__init__()
        base_distr = Uniform(low=torch.zeros(D), high=torch.ones(D))
        transforms = [SigmoidTransform().inv, AffineTransform(torch.zeros(D), torch.ones(D))]
        self.logistic_distr = TransformedDistribution(base_distr, transforms)

    def log_pdf(self, z):
        self.logistic_distr.base_dist.low = self.logistic_distr.base_dist.low.to(z.device)
        self.logistic_distr.base_dist.high = self.logistic_distr.base_dist.high.to(z.device)
        self.logistic_distr.transforms[1].loc = self.logistic_distr.transforms[1].loc.to(z.device)
        self.logistic_distr.transforms[1].scale = self.logistic_distr.transforms[1].scale.to(z.device)
        return self.logistic_distr.log_prob(z).sum(dim=1)

    def sample(self):
        return self.logistic_distr.sample()


class MLP(nn.Module):
    def __init__(self, d1, d2, num_hidden=1000):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(d1, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, d2)
                )

    def forward(self, x):
        return self.layers(x)

class additive_coupling_layer(nn.Module):
    def __init__(self, D, split_half, MLP_num_hidden=1000):
        super().__init__()
        self.D = D
        self.d_1 = D // 2
        self.d_2 = D - self.d_1
        self.m = MLP(self.d_1, self.d_2, MLP_num_hidden)
        self.split_half = split_half

    def split(self, x):
        if self.split_half == 1:
            x_i1 = x[..., :self.d_1]
            x_i2 = x[..., self.d_1:]
        else:
            x_i1 = x[..., self.d_1:]
            x_i2 = x[..., :self.d_1]
        return x_i1, x_i2

    def forward(self, x):
        x = x.clone()
        x_1, x_2 = self.split(x)
        y_1 = x_1.clone()
        y_2 = x_2 + self.m(x_1)

        if self.split_half == 0:
            y = torch.concat((y_2, y_1), dim=-1)
        else:
            y = torch.concat((y_1, y_2), dim=-1)
        return y

    def invert(self, h):
        # original formula: h_i2 = x_i2 + m(x_i1)
        # new formula:      x_i2 = h_i2 - m(h_i1)
        h = h.clone()
        h_1, h_2 = self.split(h)
        x_1 = h_1
        x_2 = h_2 - self.m(h_1)
        if self.split_half == 0:
            x = torch.concat((x_2, x_1), dim=-1)
        else:
            x = torch.concat((x_1, x_2), dim=-1)
        return x


class NICE(nn.Module):
    def __init__(self, num_coupling_layers=4, D=28*28, MLP_num_hidden=1000):
        super().__init__()
        self.latent_distr = StandardLogisticDistribution(D)
        self.S = nn.Parameter(torch.randn(D))
        self.coupling_layers_list = nn.ModuleList([additive_coupling_layer(D, i%2, MLP_num_hidden) for i in range(num_coupling_layers)])

    def forward(self, x):
        for coupling_layer in self.coupling_layers_list:
            x = coupling_layer(x)
        z = torch.exp(self.S) * x
        log_jacobian = self.S.sum()

        return z, log_jacobian

    def generate(self, z=None, device='cpu'):
        if z == None:
            z = self.latent_distr.sample().to(device)

        x = z / torch.exp(self.S)

        # invert the coupling layers
        for coupling_layer in reversed(self.coupling_layers_list):
            x = coupling_layer.invert(x)
        return x
