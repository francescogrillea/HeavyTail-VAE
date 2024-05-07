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
        raise NotImplementedError(f"Method encode not implemented in {self.__class__.__name__}")

    def generate(self, *args):
        raise NotImplementedError(f"Method generate not implemented in {self.__class__.__name__}")

    def forward(self, x, return_kl=False):
        params = self.encode(x)
        if return_kl:
            reconstructed, kl = self.generate(*params, return_kl=return_kl)
            return reconstructed, kl

        reconstructed = self.generate(*params, return_kl=return_kl)
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

    def generate(self, mu, logvar, return_kl=False):
        std = torch.exp(0.5 * logvar)
        pdf = torch.distributions.Normal(loc=mu, scale=std)
        z = pdf.rsample()
        reconstructed = self.decoder(z)

        if return_kl:
            kl = 0.5 * (std ** 2 + mu.pow(2) - 1 - torch.log(std ** 2)).sum(dim=1).mean(dim=0)
            return reconstructed, kl
        return reconstructed

class LogNormalVAE(VAE):
    pass


class InverseGaussianVAE(VAE):
    pass
