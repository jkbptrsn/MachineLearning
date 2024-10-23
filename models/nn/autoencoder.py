import numpy as np
import torch

# Adjust
import models.nn.util as util


class CAE(torch.nn.Module):
    """Conventional autoencoder neural network model.

    A configuration is a list of tuples -- each tuple defines the
        configuration of a layer in the neural network.

    Parameters
    ----------
        config_encoder: Configuration of encoder.
        config_decoder: Configuration of decoder.
    """
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

    def initialization(self) -> None:
        """Initialization before training of model."""
        self.encoder = util.construct_nn(self.config_encoder)
        self.decoder = util.construct_nn(self.config_decoder)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward function for training of model."""
        encoded = self.encoding(x)
        decoded = self.decoding(encoded)
        return decoded

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Transformation from original space to latent space."""
        return self.encoder(x)

    def decoding(self, x: np.ndarray) -> np.ndarray:
        """Transformation from latent space to original space."""
        return self.decoder(x)

    def reconstruction(self, x) -> np.ndarray:
        """Reconstruction of input."""
        return self.forward(x)


class VAE(torch.nn.Module):
    """Variational autoencoder neural network model.

    A configuration is a list of tuples -- each tuple defines the
        configuration of a layer in the neural network.

    Parameters
    ----------
        config_encoder: Configuration of encoder.
        config_decoder: Configuration of decoder.
    """
    def __init__(
            self,
            config_encoder,
            config_decoder):
        super().__init__()

        self.config_encoder = config_encoder
        self.config_decoder = config_decoder

        self.encoder = None
        self.decoder = None

        self.mean = None
        self.mean_layer = None
        self.std = None
        self.std_layer = None

        self.initialization()

    def initialization(self) -> None:
        """Initialization before training of model."""
        self.encoder = util.construct_nn(self.config_encoder)
        self.decoder = util.construct_nn(self.config_decoder)
        # Number of nodes in last layer of encoder.
        n_last = self.config_encoder[-1][1]
        # Number of nodes in first layer of decoder.
        n_first = self.config_decoder[0][0]
        # Latent mean and std layers.
        self.mean_layer = torch.nn.Linear(n_last, n_first)
        self.std_layer = torch.nn.Linear(n_last, n_first)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward function for training of model."""
        encoded = self.encoding(x)
        sampled = self.sampling(encoded)
        decoded = self.decoding(sampled)
        return decoded

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Transformation from original space to latent space."""
        return self.encoder(x)

    def decoding(self, x: np.ndarray) -> np.ndarray:
        """Transformation from latent space to original space."""
        return self.decoder(x)

    def sampling(self, x: np.ndarray) -> np.ndarray:
        self.mean = self.mean_layer(x)
        self.std = self.std_layer(x)
        z = torch.randn_like(self.std)
        return self.mean + self.std * z

    def reconstruction(self, x) -> np.ndarray:
        """Reconstruction of input."""
        encoded = self.encoding(x)
        mean = self.mean_layer(encoded)
        decoded = self.decoding(mean)
        return decoded


def setup_ae_symmetric(
        n_features: int,
        encoding_dim: int,
        n_hidden: int,
        n_nodes: int,
        node_type: str,
        activation_type: str,
        model_type: str):
    """Setup of symmetric autoencoder model.

    Parameters
    ----------
    n_features: Number of input/output features.
    encoding_dim: Number of nodes in bottleneck layer.
    n_hidden: Number of hidden layers (excluding bottleneck layer).
        Assuming symmetric network.
    n_nodes: Number of nodes in each hidden layer.
    node_type: Type of node.
        - Linear: Affine linear transformation.
    activation_type: Type of activation function.
        - ELU: Exponential linear unit function
        - LeakyReLu:
        - ReLu: Rectified linear unit function
        - Softplus: Smooth approximation of ReLU
        - Softsign:
    model_type: Conventional (CAE) or variational (VAE) autoencoder.

    Returns
    -------
    Autoencoder model.
    """
    # Configuration of encoder.
    config_encoder_ = [(n_features, n_nodes, node_type, activation_type)]
    for _ in range(n_hidden - 1):
        layer = (n_nodes, n_nodes, node_type, activation_type)
        config_encoder_.append(layer)
    # Configuration of bottleneck.
    # For VAE, the sampler is configured in the class.
    if model_type == "CAE":
        # Bottleneck layer.
        config_encoder_.append(
            (n_nodes, encoding_dim, node_type, activation_type))
    # Configuration of decoder.
    config_decoder_ = [(encoding_dim, n_nodes, node_type, activation_type)]
    for _ in range(n_hidden - 1):
        layer = (n_nodes, n_nodes, node_type, activation_type)
        config_decoder_.append(layer)
    # Output layer. TODO: No activation function!
    config_decoder_.append((n_nodes, n_features, node_type, None))
    if model_type == "CAE":
        return CAE(config_encoder_, config_decoder_)
    return VAE(config_encoder_, config_decoder_)


def load_ae_symmetric(
        n_features: int,
        encoding_dim: int,
        n_hidden: int,
        n_nodes: int,
        node_type: str,
        activation_type: str,
        model_name: str,
        model_type: str):
    """Load symmetric autoencoder model.

    Parameters
    ----------
    n_features: Number of input/output features.
    encoding_dim: Number of nodes in bottleneck layer.
    n_hidden: Number of hidden layers (excluding bottleneck layer).
        Assuming symmetric network.
    n_nodes: Number of nodes in each hidden layer.
    node_type: Type of node.
    activation_type: Type of activation function.
    model_name: Name of model (remember .pt extension).
    model_type: Conventional (CAE) or variational (VAE) autoencoder.

    Returns
    -------
    Loaded autoencoder model.
    """
    model = setup_ae_symmetric(n_features, encoding_dim, n_hidden,
                               n_nodes, node_type, activation_type, model_type)
    model.load_state_dict(torch.load(f"output/models/{model_name}",
                                     weights_only=True))
    model.eval()
    return model


def train_ae(
        model,
        loss_function,
        optimizer,
        n_epochs: int,
        data: torch.Tensor,
        model_name: str,
        model_type: str) -> None:
    """Train autoencoder model.

    Parameters
    ----------
    model: Autoencoder model.
    loss_function: Loss function.
    optimizer: Numerical optimizer.
    n_epochs: Number of epochs.
    data: Dataset.
    model_name: Name of saved model (remember .pt extension).
    model_type: Conventional (CAE) or variational (VAE) autoencoder.
    """
    for epoch in range(n_epochs):
        # Model output.
        recon = model(data)
        # Apply loss function.
        if model_type == "CAE":
            loss = loss_function(recon, data)
        elif model_type == "VAE":
            mean, std = model.mean, model.std
            loss = loss_function(recon, data, mean, std)
        else:
            raise ValueError(f"Model type {model_type} unknown.")
        # Numerical optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            sq_loss = np.sqrt(loss.item())
            print(f"Epoch: {epoch + 1:6}, "
                  f"Loss: {loss.item():11.6f}, "
                  f"Square root of loss [%]: {100 * sq_loss:8.3f}")
    # Save model.
    torch.save(model.state_dict(), f"output/models/{model_name}")
