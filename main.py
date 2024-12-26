import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from train import train, Dequantize
from NICE import NICE

train_model = True

if __name__ == "__main__":
    device = 'cpu'
    transform = transforms.Compose([ToTensor(), Dequantize()])
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(mnist_data, batch_size=32, shuffle=True)
    normalizing_flow = NICE().to(device)

    # test invertibility
    x = normalizing_flow.generate()
    x_latent, _ = normalizing_flow(x)

    x_reconstructed = normalizing_flow.generate(x_latent)
    assert torch.allclose(x, x_reconstructed,rtol=1e-04, atol=1e-06)

    try:
        normalizing_flow.load_state_dict(torch.load('normalizing_flow_weights.pth'))
    except FileNotFoundError:
        None

    # reconstructing from z without training
    x = normalizing_flow.generate(device)
    normalizing_flow.eval()
    plt.imshow(x.detach().cpu().numpy().reshape(28, 28, 1), cmap="gray")
    plt.savefig("pre-training")

    if train_model == True:
        normalizing_flow.train()
        optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.0002, weight_decay=0.9)

        loss = train(normalizing_flow, dataloader, optimizer, epochs=40, device=device)

        torch.save(normalizing_flow.state_dict(), 'normalizing_flow_weights.pth')

    normalizing_flow.eval()
    x = normalizing_flow.generate(device)
    plt.imshow(x.detach().cpu().numpy().reshape(28, 28, 1), cmap="gray")
    plt.savefig("post-training")
    plt.plot(loss)
    plt.savefig("loss")
