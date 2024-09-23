import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import util


class AE(torch.nn.Module):
    """Autoencoder."""

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
    """Variational autoencoder."""

    def __init__(
            self,
            config_encoder,
            config_decoder):
        super().__init__(config_encoder, config_decoder)


if __name__ == "__main__":

    # TODO: Why?
    torch.manual_seed(0)

    n_factors = 2

    config_encoder_ = [(8, 8, "Linear", "ReLu"),
                       (8, n_factors, "Linear", "ReLu")]

    config_decoder_ = [(n_factors, 8, "Linear", "ReLu"),
                       (8, 8, "Linear", None)]

    model = AE(config_encoder_, config_decoder_)

    criterion = torch.nn.MSELoss()

    # TODO: Different learning rate? Different weigth decay?
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1.-5)

    data_total = None

    currencies = ("DKK", "EUR", "GBP", "JPY", "USD")

    rate_years = (1, 2, 3, 5, 10, 15, 20, 30)

    column_names = list()
    for year in rate_years:
        column_names.append(f" {year}Y Par Swap Rate")

    for cur in currencies:

        if cur == "DKK":
            data = pd.read_excel(f"data/{cur} swaps.xlsx", index_col=0)
        else:
            data = pd.read_excel(f"data/{cur} swaps.xls", index_col=0)

        col_names = []
        for n in column_names:
            col_names.append(cur + n)

        data = data[col_names]

        if data_total is None:
            data_total = data.values
        else:
            data_total = np.vstack([data_total, data.values])

    print(data_total.shape)

    data_total = torch.from_numpy(data_total)
    # TODO: Double type?
    data_total = data_total.to(torch.float32)

    num_epochs = 25001

    output = None

    for epoch in range(num_epochs):
        recon = model(data_total)
        loss = criterion(recon, data_total)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
        if epoch == num_epochs - 1:
            output = (epoch, recon.detach().numpy())

    curve_ids = ("2010-01-29", "2017-07-31", "2019-08-30", "2023-02-28", "2023-12-29")

    n_curves = len(curve_ids)

    color = iter(plt.cm.gist_rainbow(np.linspace(0, 1, n_curves)))

    if currencies[0] == "DKK":
        data_pd = pd.read_excel(f"data/{currencies[0]} swaps.xlsx", index_col=0)
    else:
        data_pd = pd.read_excel(f"data/{currencies[0]} swaps.xls", index_col=0)

    for n, idx in enumerate(data_pd.index):
        if idx in curve_ids:
            c = next(color)
            plt.plot(rate_years, data_total[n, :],
                     color=c, linestyle="-", marker="o", label=idx)
            plt.plot(rate_years, output[1][n, :],
                     color=c, linestyle="--", marker="x")
    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()
