import torch
import torch.nn as nn
from base_distributions import StandardNormalDistribution
from neural_nets import ResNet

class AffineCouplingLayer(nn.Module):
    def __init__(self, split_half, masking_scheme, resnet_hidden_channels=64):
        super().__init__()
        self.split_half = split_half # 0 or 1
        self.s = ResNet(in_channels=3, hidden_channels=resnet_hidden_channels, final_activation='scaled_tanh')
        self.t = ResNet(in_channels=3, hidden_channels=resnet_hidden_channels)
        self.masking_scheme = masking_scheme # 'channelwise' or 'checkerboard'

    def checkerboard_mask(self, x):
        # Generates a checkerboard style mask b as seen in figure 3 & section 3.4 of paper
        b = torch.ones_like(x)

        # for each channel, zero out every other value of every other row
        b[:, :, ::2, (self.split_half)::2] = 0

        # for each channel, zero out every other value of every other row, both shifted by 1
        b[:, :, 1::2, (1-self.split_half)::2] = 0

        return b

    def channelwise_mask(self, x):
        # Generates a channelwise style mask b as seen in figure 3 & section 3.4 of paper
        b = torch.ones_like(x)

        # zero out every other channel, starting based on the split half of this layer
        b[:, (self.split_half)::2, :, :] = 0

        return b

    def forward(self, x):
        b = checkerboard_mask(x) if self.masking_scheme is 'checkerboard' else channelwise_mask(x)

        # equation 9 of paper
        y = (b * x) + (1 - b) * (x * torch.exp(self.s(b * x)) + self.t(b * x))
        return y

class RealNVP_Level(nn.Module):
    def __init__(self, n_components, start_mask, resnet_hidden_channels=64):
        super().__init__()

        self.checkerboard_masking_layers = nn.Sequential(
            AffineCouplingLayer(start_mask, 'checkerboard', resnet_hidden_channels),
            AffineCouplingLayer(1-start_mask, 'checkerboard', resnet_hidden_channels),
            AffineCouplingLayer(start_mask, 'checkerboard', resnet_hidden_channels),
            )

        self.channelwise_masking_layers = nn.Sequential(
            AffineCouplingLayer(start_mask, 'channelwise', resnet_hidden_channels),
            AffineCouplingLayer(1-start_mask, 'channelwise', resnet_hidden_channels),
            AffineCouplingLayer(start_mask, 'channelwise', resnet_hidden_channels),
            )

    def squeeze(self, x):
        # see section 3.6 of paper
        n = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        assert h % 2 == w % 2 == 0

        return x.view(n, 4*c, h/2, w/2)

    def forward(self, x):
        h = self.checkerboard_masking_layers(x)
        h = self.squeeze(h)
        h = channelwise_masking_layers(h)
        return h

class RealNVP(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_distr = StandardNormalDistribution() # needs some kind of dimension: How do 2d inputs get mapped to a standard normal?



