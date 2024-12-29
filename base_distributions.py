import torch
import torch.nn as nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transforms import AffineTransform


# --------------------- NICE std logistic distribution ---------------- ###

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
