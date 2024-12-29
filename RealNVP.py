import torch
import torch.nn as nn
from base_distributions import StandardNormalDistribution

class AffineCouplingLayer(nn.Module):
    def __init__(self, D, split_half, resnet_hidden_channels=64):
        super().__init__()
        self.D = D
        self.d_1 = D // 2
        self.d_2 = D - self.d_1
        self.split_half = split_half
        self.s = ResNet(in_channels=3, hidden_channels=resnet_hidden_channels)
        self.t = ResNet(in_channels=3, hidden_channels=resnet_hidden_channels)

    def checkerboard_mask(self, x):
        # TODO: implement checkerboard style masking as seen in paper
        pass
    def channelwise_mask(self, x):
        # TODO: implement channel-wise masking as seen in paper
        pass

    def forward(self, x):
        # TODO: mask, return out = s(masked_input) * (other_mask) + t(masked_input)  
        pass

class RealNVP(nn.Module):
    super().__init__()
    self.base_distr = StandardNormalDistribution() # needs some kind of dimension: How do 2d inputs get mapped to a standard normal?


