import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from NICE import NICE

def train(normalizing_flow, dataloader, optimizer, epochs=1):
    training_loss = []
    for _ in range(epochs):
        for x_batch in dataloader:
            z, log_jacobian = normalizing_flow(x_batch)
            log_likelihood = normalizing_flow.latent_distribution.log_pdf(z) + log_jacobian
            loss = -log_likelihood.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
    return training_loss

class Dequantize:
    def __call__(self, tensor, corruption_level=1):
        noise = corruption_level * torch.rand_like(tensor)
        tensor += noise
        tensor /= (255 + corruption_level)
        return tensor

if __name__ == "__main__":
    transform = transforms.Compose([ToTensor(), Dequantize()])
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(mnist_data, batch_size=32, shuffle=True)
    normalizing_flow = NICE()

    optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.0002, weight_decay=0.9)

    loss = train(normalizing_flow, dataloader, optimizer)

    print(loss)
