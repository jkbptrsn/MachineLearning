import torch

# Adjust
import models.nn.util as util


class AE(torch.nn.Module):
    """Autoencoder neural network model.

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

    def initialization(self):
        self.encoder = util.construct_nn(self.config_encoder)
        self.decoder = util.construct_nn(self.config_decoder)

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


def setup_ae_symmetric(
        n_features: int,
        encoding_dim: int,
        n_hidden: int,
        n_nodes: int,
        node_type: str,
        activation_type: str):
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
    Returns
    -------
    Autoencoder model
    """
    # Configuration of encoder.
    config_encoder_ = [(n_features, n_nodes, node_type, activation_type)]
    for _ in range(n_hidden - 1):
        layer = (n_nodes, n_nodes, node_type, activation_type)
        config_encoder_.append(layer)
    config_encoder_.append((n_nodes, encoding_dim, node_type, activation_type))
    # Configuration of decoder.
    config_decoder_ = [(encoding_dim, n_nodes, node_type, activation_type)]
    for _ in range(n_hidden - 1):
        layer = (n_nodes, n_nodes, node_type, activation_type)
        config_decoder_.append(layer)
    # TODO: Why not apply activation function at output layer?
    config_decoder_.append((n_nodes, n_features, node_type, None))
    return AE(config_encoder_, config_decoder_)


def load_ae_symmetric(
        n_features: int,
        encoding_dim: int,
        n_hidden: int,
        n_nodes: int,
        node_type: str,
        activation_type: str,
        model_name: str):
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

    Returns
    -------
    Loaded model.
    """
    model = setup_ae_symmetric(n_features, encoding_dim, n_hidden,
                               n_nodes, node_type, activation_type)
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
        model_name: str):
    """Train autoencoder model.

    TODO: Check the sequence of operations.

    Parameters
    ----------
    model: Autoencoder model.
    loss_function: Loss function.
    optimizer: Numerical optimizer.
    n_epochs: Number of epochs.
    data: Dataset.
    model_name: Name of saved model (remember .pt extension).
    """
    for epoch in range(n_epochs):
        # Model output.
        recon = model(data)
        # Apply loss function.
        loss = loss_function(recon, data)
        # Numerical optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
    # Save model.
    torch.save(model.state_dict(), f"output/models/{model_name}")
