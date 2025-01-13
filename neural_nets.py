import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d1, d2, num_hidden=1000):
        super().__init__()
        # NICE paper calls for 5 layers of 1000 neurons each
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

# ResNets modeled after the torchvision implementation: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1 ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x should be NxCxHxW
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity
        out = self.relu(out)

class ResNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_blocks=8, final_activation='none'):
        super().__init__()
        self.final_activation = final_activation

        self.blocks = nn.ModuleList([
            [ResNetBlock(in_channels, hidden_channels)] + # first block
            [ResNetBlock(hidden_channels, hidden_channels) for _ in range(n_blocks - 2)] + # All blocks that aren't first or last are 'middle blocks'
            [ResNetBlock(hidden_channels, in_channels)] # last block
            ])

        # TODO: learned scale
        self.learned_scale = nn.Parameter(torch.tensor())

    def forward(self, x):
        for residual_block in self.blocks:
            x = residual_block(x)

        if self.final_activation == 'scaled_tanh':
            x = self.learned_scale * torch.tanh(x)

        return x
