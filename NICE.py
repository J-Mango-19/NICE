import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transforms import AffineTransform

# D is the dimension of the data, preserved through all transformations
class StandardLogisticDistribution:
    def __init__(self, D):
        base_distr = Uniform(low=torch.zeros(D), high=torch.ones(D))
        transforms = [SigmoidTransform().inv, AffineTransform(loc=0, scale=1)]
        self.logistic_distr = TransformedDistribution(base_distr, transforms)

    def log_pdf(self, z):
        return self.m.log_prob(z).sum(dim=1)

    def sample(self):
        return self.logistic_distr.sample()


class MLP(nn.Module):
    def __init__(self, d1, d2):
        self.layers = nn.Sequential(
                nn.Linear(d1, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, d2)
                )

    def forward(self, x):
        return self.layers(x)

class additive_coupling_layer(nn.Module):
    def __init__(self, D, split):
        super().__init__()

    self.m = MLP()

    def split(self, x):
        if self.split == 1:
            x_i1 = x[:d]
            x_i2 = x[d:]
        else:
            x_i1 = x[d:]
            x_i2 = x[:d]
        return x_i1, x_i2

    def forward(self, x):
        x_1, x_2 = self.split(x)
        y_1 = x_1.clone()
        y_2 = x_2 + self.m(x_1)

        y = torch.concat(y1, y2)
        return y

    def invert(self, h):
        # original formula: h_i2 = x_i2 + m(x_i1)
        # new formula:      x_i2 = h_i2 - m(h_i1)
        h_i1, h_i2 = self.split(h)
        x_i1 = h_i1
        x_i2 = h_i2 - self.m(h_i1)
        x = torch.concat(x_i1, x_i2)
        return x


class NICE(nn.Module):
    def __init__(self, num_coupling_layers=4, D=28*28):
        super().__init__()
        self.latent_distr = StandardLogisticDistribution(D)
        self.S = nn.Parameter(torch.randn(D))
        coupling_layers_list = nn.ModuleList([additive_coupling_layer(D, i%2) for i in range(num_coupling_layers)])

    def forward(self, x):
        for coupling_layer in self.coupling_layers_list:
            x = coupling_layer(x)
        z = self.S.exp() * x
        log_jacobian = self.S.sum()

        return z, log_jacobian

    def generate(self):
        z = self.latent_distr.sample()
        x = z / self.S.exp()

        # invert the coupling layers
        for coupling_layer in reversed(self.coupling_layers_list):
            x = coupling_layers.invert(x)
        return x
