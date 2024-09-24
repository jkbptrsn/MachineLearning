import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import models.nn.autoencoder as ae


def load_dataframe(
        currency: str,
        tenors: tuple) -> pd.DataFrame:
    """Load curve data for currency into DataFrame.

    Parameters
    ----------
    currency: Currency to include in dataset.
    tenors: Tenors to include in dataset.

    Returns
    -------
    Dataset
    """
    # Relevant column names.
    column_names = list()
    for year in tenors:
        column_names.append(f" {year}Y Par Swap Rate")
    if currency == "DKK":
        data = pd.read_excel(f"data/{currency} swaps.xlsx", index_col=0)
    else:
        data = pd.read_excel(f"data/{currency} swaps.xls", index_col=0)
    # Get relevant columns.
    mask = []
    for name in column_names:
        mask.append(currency + name)
    return data[mask]


def setup_dataset(
        currencies: tuple,
        tenors: tuple) -> torch.Tensor:
    """Setup dataset...

    Parameters
    ----------
    currencies: Currencies to include in dataset.
    tenors: Tenors to include in dataset.

    Returns
    -------
    Dataset.
    """
    data = None
    for cur in currencies:
        data_tmp = load_dataframe(cur, tenors)
        if data is None:
            data = data_tmp.values
        else:
            data = np.vstack([data, data_tmp.values])
    # Convert to torch.Tensor.
    data = torch.from_numpy(data)
    # TODO: Double type?
    data = data.to(torch.float32)
    return data


if __name__ == "__main__":

    # ...
    torch.manual_seed(3)

    # Choose number of factors.
    n_factors = 2

    # Choose currencies.
    currencies = ("DKK", "EUR", "GBP", "JPY", "USD")

    # Choose tenors.
    tenors = (1, 2, 3, 5, 10, 15, 20, 30)

    # Dataset.
    data = setup_dataset(currencies, tenors)
    print(data.shape)

    # Model parameters.
    n_features = len(tenors)
    encoding_dim = n_factors
    n_hidden = 1
    n_nodes = 8
    node_type = "Linear"
    activation_type = "ReLu"

    # Model name.
    model_name = "ae_1_new.pt"

    # Train model.
    if True:

        # Number of training epochs.
        n_epochs = 25001

        # Setup model.
        model = ae.setup_ae_symmetric(
            n_features,
            encoding_dim,
            n_hidden,
            n_nodes,
            node_type,
            activation_type)

        # Loss function.
        loss_function = torch.nn.MSELoss()

        # Numerical optimizer.
        # TODO: Different learning rate? Different weight decay?
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        ae.train_ae(model,
                    loss_function,
                    optimizer,
                    n_epochs,
                    data,
                    model_name)

    # Plot reconstructed rate curves.
    if False:

        # Load model.
        model = ae.load_ae_symmetric(
                n_features,
                encoding_dim,
                n_hidden,
                n_nodes,
                node_type,
                activation_type,
                model_name)

        curve_ids = ("01/29/2010", "07/31/2017", "08/30/2019", "02/28/2023", "12/29/2023")

        n_curves = len(curve_ids)

        color = iter(plt.cm.gist_rainbow(np.linspace(0, 1, n_curves)))

        data = load_dataframe("DKK", tenors)

        for n, idx in enumerate(data.index):
            if idx in curve_ids:

                data_tmp = torch.from_numpy(data.iloc[n].values)
                data_tmp = data_tmp.to(torch.float32)
                data_tmp = model.forward(data_tmp).detach().numpy()

                c = next(color)
                plt.plot(tenors, data.values[n, :],
                         color=c, linestyle="-", marker="o", label=idx)
                plt.plot(tenors, data_tmp,
                         color=c, linestyle="--", marker="x")
        plt.xlabel("Time")
        plt.ylabel("Rate")
        plt.legend()
        plt.show()
