import torch


class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.LeakyReLU(),
        )

        self.latent_dim = latent_dim

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, input_dim),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        pass

    def generate(self, *args):
        pass

    def forward(self, x):
        params = self.encode(x)
        reconstructed = self.generate(*params)

        return reconstructed


class GaussianVAE(VAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mu_head = torch.nn.Linear(300, self.latent_dim)
        self.logvar_head = torch.nn.Linear(300, self.latent_dim)

    def encode(self, x):
        x = self.encoder(x)

        mu = self.mu_head(x)
        logvar = self.logvar_head(x)

        return mu, logvar

    def generate(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        p = torch.randn(1)
        z = mu + p * std

        return self.decoder(z)


class LogNormalVAE(VAE):
    pass


class InverseGaussianVAE(VAE):
    pass
