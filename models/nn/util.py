from collections import OrderedDict

import torch


def construct_nn(
        configuration: list,
        description: str = "") -> torch.nn.Sequential:
    """Construct neural network based on configuration.

    Parameters
    ----------
    configuration: List of tuples -- each tuple defines the
        configuration of a layer in the neural network.
        * Type of layer:
            - Linear: Affine linear transformation.
        * Type of activation function:
            - ELU: Exponential linear unit function
            - LeakyReLu:
            - ReLu: Rectified linear unit function
            - Softplus: Smooth approximation of ReLU
            - Softsign:
    description: String describing the neural network.

    Returns
    -------
    Container with all layers in the neural network.
    """
    layers = OrderedDict()
    idx_start = 1
    for idx, (n_nodes_in,
              n_nodes_out,
              node_type,
              activation_type) in enumerate(configuration):
        # Add layer.
        key = f"{description}: layer {idx + idx_start}"
        if node_type == "Linear":
            layers[key] = torch.nn.Linear(n_nodes_in, n_nodes_out)
        else:
            raise ValueError("Node type is unknown.")
        # Apply activation function.
        if activation_type is not None:
            key = f"{description}: activation {idx + idx_start}"
            if activation_type == "ELU":
                layers[key] = torch.nn.ELU()
            elif activation_type == "ReLu":
                layers[key] = torch.nn.ReLU()
            elif activation_type == "LeakyReLu":
                layers[key] = torch.nn.LeakyReLU()
            elif activation_type == "Softplus":
                layers[key] = torch.nn.Softplus()
            elif activation_type == "Softsign":
                layers[key] = torch.nn.Softsign()
            else:
                raise ValueError("Activation type is unknown.")
    return torch.nn.Sequential(layers)