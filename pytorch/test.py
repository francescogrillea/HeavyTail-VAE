import sys

import torch
import matplotlib.pyplot as plt

from VAE import GaussianVAE


device = "cuda" if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    path = 'vae.bin' if len(sys.argv) == 1 else sys.argv[1]
    model = GaussianVAE(784, 2)
    model.load_state_dict(torch.load(path))

    N = 5

    fig, ax = plt.subplots(N, N, figsize=(N, N))

    for i in range(N):
        for j in range(N):
            mu = torch.randn(1, 2).to(device)
            std = torch.randn(1, 2).to(device).abs()
            recostructed = model.generate(mu, std).cpu().detach().numpy().reshape(28, 28)
            ax[i][j].imshow(recostructed, cmap="gray")
            ax[i][j].axis("off")

    plt.show()
