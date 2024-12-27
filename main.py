import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from NICE import NICE
from train import train, Dequantize
from utils import parse_args, load_model, plot_random_samples
from tests import test_invertibility

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transform = transforms.Compose([ToTensor(), Dequantize()])
    mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist_data, batch_size=32, shuffle=True)

    nice_model = NICE(args.num_coupling_layers, 28*28, args.MLP_num_hidden).to(device)

    # pre training invertibility check
    test_invertibility(nice_model, device)

    load_model(nice_model, 'normalizing_flow_weights.pth', device)

    if args.train == True:
        nice_model.train()
        optimizer = torch.optim.Adam(nice_model.parameters(), lr=args.lr, weight_decay=args.wd)
        loss = train(nice_model, dataloader, optimizer, epochs=args.epochs, device=device)
        torch.save(nice_model.state_dict(), 'nice_model_weights.pth')
        plt.plot(loss)
        plt.savefig("assets/loss.png")

    plot_random_samples(nice_model, device)

