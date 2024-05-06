import sys

import torch
import torchvision
from tqdm import tqdm

from VAE import GaussianVAE


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_mnist = torchvision.datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
)

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3


if __name__ == "__main__":
    model = GaussianVAE(784, 2).to(device)

    dl = torch.utils.data.DataLoader(train_mnist, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        bar = tqdm(dl)
        for digit, _ in bar:
            digit = digit.to(device)
            digit = digit.view(digit.shape[0], 28 * 28)
            recostructed = model(digit)
            error = loss_fn(recostructed, digit)

            loss = error + 0.005 # * kl
            # bar.set_description(f"Loss: {loss.item():.2f} | Error: {error.item():.2f} | KL: {kl.item():.2}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    path = '' if len(sys.argv) == 1 else sys.argv[1]
    torch.save(model.state_dict(), '../../Progetto/vae.bin')
