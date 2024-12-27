import torch

def test_invertibility(normalizing_flow, device):
    x = normalizing_flow.generate(device=device)
    x_latent, _ = normalizing_flow(x)
    x_reconstructed = normalizing_flow.generate(x_latent)
    assert torch.allclose(x, x_reconstructed,rtol=1e-04, atol=1e-06)

