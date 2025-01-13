import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from NICE import NICE

def train(normalizing_flow, dataloader, optimizer, epochs=1, device='cpu'):
    print("Training... ")
    normalizing_flow.train()
    training_loss = []
    normalizing_flow.latent_distr.to(device)
    for epoch in range(epochs):
        for x_batch, _ in dataloader:
            x_batch = x_batch.view(-1, 28*28).to(device)
            z, log_jacobian = normalizing_flow(x_batch)
            log_likelihood = normalizing_flow.latent_distr.log_pdf(z) + log_jacobian
            loss = -log_likelihood.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
        print(f'Loss at end of epoch {epoch}: {loss.item():.3f}')
    return training_loss

class Dequantize:
    def __call__(self, tensor, corruption_level=1.0):
        tensor *= 255.0
        noise = corruption_level * torch.rand_like(tensor)
        tensor += noise
        tensor /= (255.0 + corruption_level)
        return tensor
