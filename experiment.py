import sys
import torch
import matplotlib.pyplot as plt

from NICE import NICE
from utils import load_model

def get_influential_idxs(num_requested_idxs, model):
    """
    Returns the latent indices with the lowest S_ii coefficients
    Lower S_ii coefficients means the model learned to make the base distribtution more
    sensitive to changes at these indices, therefore they are more influential
    """

    # map every s value to its index in the original s vector
    dims_dict = {s.item() : i for i, s in enumerate(list(model.S.exp()))}

    # sort the s values
    sorted_s = sorted([s.item() for s in list(model.S.exp())])

    # map the lowest of the sorted s values to their indices
    influential_idxs = [dims_dict[s] for s in sorted_s[:num_requested_idxs]]
    return influential_idxs

def traverse_latent_space_multi_img(model, latent_traversal_idx, traversal_step_size=2, num_base_imgs=4, num_traversed_imgs=4, device='cpu'):
    """
    Incrementally increases ("traverses") a single index of several images' latent vectors, plotting the reconstructed images at each increase
    """
    # generate a handful of images
    # reconstruct the images' latent spaces as they're increased along the specified index
    model.eval()
    with torch.no_grad():
        imgs = [model.generate(device=device) for _ in range(num_base_imgs)]

        rows, cols = num_base_imgs, num_traversed_imgs + 1
        figure = plt.figure(figsize=(3*num_traversed_imgs, 3*num_base_imgs))

        for i in range(rows):
            # plot original images
            img_i_latent, _ = model(imgs[i])
            figure.add_subplot(rows, cols, i*cols + 1)
            plt.imshow(imgs[i].cpu().view(28,28).clip(0,1), cmap="gray")
            if i == 0: plt.title('Original Img', fontsize=25)

            # plot latent-traversed reconstructions
            for j in range(2, cols+1):
                img_i_latent[latent_traversal_idx] += traversal_step_size
                ax = figure.add_subplot(rows, cols, i*cols + j)
                new_img_i = model.generate(img_i_latent, device=device)
                assert new_img_i.shape == torch.Size([784])
                plt.imshow(new_img_i.cpu().view(28, 28).clip(0,1), cmap="gray")
                if i == 0 and j == 4:
                    plt.title(f'adding {traversal_step_size} to latent vector at index {latent_traversal_idx}', fontsize=25)

        for ax in figure.axes:
            ax.axis('off')

        # save the figure
        file_name = './assets/latent_traversals_multi_img.png'
        print(f"saving to {file_name}")
        plt.savefig(file_name)

def traverse_latent_space_multi_idx(model, traversal_step_size=3, num_idxs=3, num_to_generate=5, device='cpu'):
    """
    Plots reconstructions of a single image as its latent vector is incrementally increased at a handful of indices
    """

    # get the `num_idxs` most influential indices
    idxs = get_influential_idxs(num_idxs, model)
    model.eval()
    with torch.no_grad():
        orig_img = model.generate(device=device)

        latent_vectors = [model(orig_img)[0] for _ in range(num_idxs)]

        rows, cols = num_idxs, num_to_generate + 1
        figure = plt.figure(figsize=(3*num_to_generate, 3*num_idxs))

        for i in range(rows):
            # plot original images
            figure.add_subplot(rows, cols, i*cols + 1)
            plt.imshow(orig_img.cpu().view(28, 28).clip(0,1), cmap='gray')
            if i == 0:
                plt.title('Original Img', fontsize=25)

            cur_latent_idx = idxs[i]

            # plot latent-traversed reconstructions
            for j in range(2, cols + 1):
                latent_vectors[i][cur_latent_idx] += traversal_step_size
                new_img = model.generate(latent_vectors[i], device=device)

                ax = figure.add_subplot(rows, cols, i*cols + j)
                assert new_img.shape == torch.Size([784])
                plt.imshow(new_img.cpu().view(28, 28).clip(0,1), cmap="gray")
                if i == 0 and j == 4:
                    plt.title(f'adding {traversal_step_size} to latent vector at indices {','.join(map(str, idxs))}', fontsize=25)

        for ax in figure.axes:
            ax.axis('off')

        # save the figure
        file_name = './assets/latent_traversals_multi_idx.png'
        print(f"saving to {file_name}")
        plt.savefig(file_name)

if __name__ == '__main__':
    device = torch.device('cpu')
    nice_model = NICE()
    if not load_model(nice_model, 'normalizing_flow_weights.pth', device):
        print("Could not load trained model to perform experiments on, exiting...")
        sys.exit(1)

    # Traverse one latent dimension for several images; save reconstructions
    _, idx = get_influential_idxs(2, nice_model)
    traverse_latent_space_multi_img(nice_model, idx, device=device)

    # Traverse several latent dimensions for one image; save reconstructions
    traverse_latent_space_multi_idx(nice_model, device=device)
