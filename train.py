import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from NICE import NICE

def train(normalizing_flow, dataloader, optimizer, epochs=1, device='cpu'):
    normalizing_flow.train()
    training_loss = []
    normalizing_flow.latent_distr.to(device)
    for epoch in range(epochs):
        for x_batch, _ in dataloader:
            x_batch = x_batch.view(-1, 28*28).to(device)
            z, log_jacobian = normalizing_flow(x_batch)
            log_likelihood = normalizing_flow.latent_distr.log_pdf(z) + log_jacobian
            loss = -log_likelihood.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
        print(f'Loss at end of epoch {epoch}: {loss.item()}')
    return training_loss

class Dequantize:
    def __call__(self, tensor, corruption_level=1.0):
        tensor *= 255.0
        noise = corruption_level * torch.rand_like(tensor)
        tensor += noise
        tensor /= (255.0 + corruption_level)
        return tensor

if __name__ == "__main__":
    transform = transforms.Compose([ToTensor(), Dequantize()])
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(mnist_data, batch_size=32, shuffle=True)
    normalizing_flow = NICE()

    # reconstructing from z without training
    x = normalizing_flow.generate()
    plt.imshow(x.detach().cpu().numpy().reshape(28, 28, 1), cmap="gray")
    plt.savefig("pre-training")

    optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.001, weight_decay=0.9)

    loss = train(normalizing_flow, dataloader, optimizer)

    x = normalizing_flow.generate()
    plt.imshow(x.detach().cpu().numpy().reshape(28, 28, 1), cmap="gray")
    plt.savefig("post-training")


