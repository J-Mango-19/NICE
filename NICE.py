import torch
import torch.nn as nn

from neural_nets import MLP
from base_distributions import StandardLogisticDistribution

class AdditiveCouplingLayer(nn.Module):
    def __init__(self, D, layer_partition, MLP_num_hidden=1000):
        super().__init__()
        # D and d defined same way as in section 3.2 of paper
        # D is the data dimension (eg D=28*28 for mnist)
        # d is half of the data dimension (eg d=28*28/2 for mnist)
        self.D = D
        self.d = D // 2

        # m is the coupling function, defined same way as in section 3.2 of paper
        self.m = MLP(self.d, (self.D - self.d), MLP_num_hidden)

        # a layer's layer_partition determines its partition of x
        self.layer_partition = layer_partition

    def split(self, x):
        # Determine which half of x to treat as x_i1, x_i2 based on the layer partition
        if self.layer_partition == 1:
            x_i1 = x[..., :self.d]
            x_i2 = x[..., self.d:]
        else:
            x_i1 = x[..., self.d:]
            x_i2 = x[..., :self.d]
        return x_i1, x_i2

    def concatenate_in_order(self, x_i1, x_i2):
        # Regardless of which half we treat as x_i1 and x_i2, dimensions must remain in order
        if self.layer_partition == 1:
            x = torch.concat((x_i1, x_i2), dim=-1)
        else:
            x = torch.concat((x_i2, x_i1), dim=-1)
        return x

    def forward(self, x):
        print(f'mean of incoming x in coupling layer forward pass: {x.mean()}')
        # Alternate (by coupling layer) which half of the data is fed to the coupling fxn m
        x_i1, x_i2 = self.split(x) # x_i1, x_i2 defined in same way as in section 3.2 of paper

        # identity
        y_i1 = x_i1

        # additive coupling law
        y_i2 = x_i2 + self.m(x_i1)

        y = self.concatenate_in_order(y_i1, y_i2)

        print(f'mean of outgoing y in coupling layer forward pass: {y.mean()}')
        return y
        """
        if self.layer_partition == 1:
            y = torch.concat((y_1, y_2), dim=-1)
        else:
            y = torch.concat((y_2, y_1), dim=-1)
        return y
        """

    def invert(self, h):
        # forward pass:         h_i2 = x_i2 + m(x_i1)
        # inverse forward pass: x_i2 = h_i2 - m(h_i1)
        h_i1, h_i2 = self.split(h)

        # identity 
        x_i1 = h_i1

        # inverse of additive coupling law
        x_i2 = h_i2 - self.m(h_i1)

        x = self.concatenate_in_order(x_i1, x_i2)
        return x
        """
        # x must be concatenated in the original order of x
        if self.layer_partition == 0:
            x = torch.concat((x_2, x_1), dim=-1)
        else:
            x = torch.concat((x_1, x_2), dim=-1)
        return x
        """


class NICE(nn.Module):
    def __init__(self, num_coupling_layers=4, D=28*28, MLP_num_hidden=1000):
        super().__init__()
        self.latent_distr = StandardLogisticDistribution(D)
        self.S = nn.Parameter(torch.ones(D)) # vector containing the diagonal elements of diagonal scaling matrix S (section 3.3 of paper)
        self.coupling_layers_list = nn.ModuleList(
                [AdditiveCouplingLayer(D, i%2, MLP_num_hidden) for i in range(num_coupling_layers)]
        )

    def forward(self, x):
        for coupling_layer in self.coupling_layers_list:
            x = coupling_layer(x)
        z = torch.exp(self.S) * x
        log_jacobian = self.S.sum() # Adhere to change of variables formula (all other transformations have unit Jacobian)
        return z, log_jacobian

    def generate(self, z=None, device='cpu'):
        if z == None:
            z = self.latent_distr.sample().to(device)

        # invert the scaling layer
        x = z / torch.exp(self.S)

        # invert the coupling layers
        for coupling_layer in reversed(self.coupling_layers_list):
            x = coupling_layer.invert(x)
        return x
