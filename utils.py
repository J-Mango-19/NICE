import argparse
import torch
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=10**-3) # was 0.0002
    parser.add_argument("--beta_2", type=float, default=0.1)
    parser.add_argument("--beta_1", type=float, default=0.9)
    #parser.add_argument("--eps", type=float, default=10e-4)
    parser.add_argument("--eps", type=float, default=10**-2)
    parser.add_argument("--wd", type=float, default=0) # TODO: remove. Was 0.9
    parser.add_argument("--num_coupling_layers", type=int, default=4)
    parser.add_argument("--MLP_num_hidden", type=int, default=1000)
    parser.add_argument("--load_model", type=bool, default=True)
    args = parser.parse_args()
    return args


def load_model(model, filename, device):
    try:
        model.load_state_dict(torch.load(filename, weights_only=True, map_location=device))
        print(f"loaded model from {filename}")
        return True
    except FileNotFoundError:
        print(f"Failed to load model from {filename}")
        return False


def plot_random_samples(normalizing_flow, device, png_name='nf_samples'):
    normalizing_flow.eval()
    num_imgs = 10
    fig, axs = plt.subplots(num_imgs, num_imgs, figsize=(10, 10))
    for i in range(num_imgs):
        for j in range(num_imgs):
            x = normalizing_flow.generate(device=device).unsqueeze(0).data.cpu().numpy().clip(0,1)
            axs[i, j].imshow(x.reshape(28, 28), cmap='gray')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    print(f'saving image samples in assets/{png_name}.png')
    plt.savefig(f'assets/{png_name}.png')
