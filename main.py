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

    # load .pth weights if possible
    """
    model_load_success = load_model(nice_model, 'normalizing_flow_weights.pth', device)
    if model_load_success and not args.train:
        plot_random_samples(nice_model, device)
    """

    # after loaded model invertibility check
    test_invertibility(nice_model, device)

    if args.train == True:
        optimizer = torch.optim.Adam(nice_model.parameters(), lr=args.lr, betas=(0.9, args.beta_2), eps=args.eps, weight_decay=args.wd)
        loss = train(nice_model, dataloader, optimizer, epochs=args.epochs, device=device)

        # save weights
        #torch.save(nice_model.state_dict(), 'nice_model_weights.pth')

        # save loss plot and randomly generated samples
        plt.plot(loss)
        plt.savefig("assets/loss.png")
        plot_random_samples(nice_model, device)


