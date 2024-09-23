import torch

import util


class AE(torch.nn.Module):
    """Autoencoder neural network model."""

    def __init__(
            self,
            config_encoder,
            config_decoder):
        super().__init__()

        self.config_encoder = config_encoder
        self.config_decoder = config_decoder

        self.encoder = None
        self.decoder = None

        self.initialization()

    def initialization(self):
        self.encoder = (
            util.construct_nn(self.config_encoder, description="encoder"))
        self.decoder = (
            util.construct_nn(self.config_decoder, description="decoder"))

    def forward(self, x):
        encoded = self.encoding(x)
        decoded = self.decoding(encoded)
        return decoded

    def encoding(self, x):
        return self.encoder(x)

    def decoding(self, x):
        return self.decoder(x)


class VAE(AE):
    """Variational autoencoder neural network model."""

    def __init__(
            self,
            config_encoder,
            config_decoder):
        super().__init__(config_encoder, config_decoder)
