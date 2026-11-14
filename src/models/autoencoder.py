"""Deep Reconstruction Autoencoder in PyTorch."""
import torch
import torch.nn as nn

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 11, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.Tanh(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def get_reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            mse = torch.mean((x - recon) ** 2, dim=-1)
        return mse
