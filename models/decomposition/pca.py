import numpy as np


class PCA:
    """Principal Component Analysis class.

    Parameters
    ----------
        data: np.ndarray of size (m x N), where
            - m is the number of features
            - N is the number of points in the dataset.
    """
    def __init__(
            self,
            data: np.ndarray):
        self.data = data

        # Number of features.
        self.n_features = self.data.shape[0]
        # Number of points.
        self.n_points = self.data.shape[1]
        # Mean of sample for each feature: Size (m x 1).
        self.mean = self.data.mean(axis=1).reshape((-1, 1))

        # Covariance matrix: Size (m x m).
        self.cov = None

        # Eigen values of covariance matrix: Size (m x 1) TODO?.
        self.eigen_values = None
        # Eigen vectors of covariance matrix: Size (m x m).
        self.eigen_vectors = None

    def initialization(self, subtract_mean: bool = True) -> None:
        """Initialization before calling encoder/decoder."""
        self.diagonalization(subtract_mean)

    def covariance(self, subtract_mean: bool = True) -> None:
        """Calculate covariance matrix."""
        if self.cov is None:
            if subtract_mean:
                data_hat = self.data - self.mean
            else:
                data_hat = self.data
            self.cov = np.matmul(data_hat, data_hat.transpose())
            self.cov /= self.n_points

    def diagonalization(self, subtract_mean: bool = True) -> None:
        """Diagonalization of covariance matrix."""
        self.covariance(subtract_mean)
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(self.cov)
        # Ordered according to descending eigenvalues.
        self.eigen_values = np.flip(self.eigen_values)
        self.eigen_vectors = np.flip(self.eigen_vectors, axis=1)

    def encoding(self, x: np.ndarray) -> np.ndarray:
        """Representation of x using principal component basis."""
        return np.matmul(self.eigen_vectors.transpose(), x - self.mean)

    @staticmethod
    def dimension_reduction(
            x: np.ndarray,
            n_factors: int) -> np.ndarray:
        """n_factors-dimensional representation of x."""
        x[n_factors:] = 0

        # TODO: Renormalization?

        return x

    def decoding(self, x: np.ndarray) -> np.ndarray:
        """Inverse of encoding transformation."""
        return np.matmul(self.eigen_vectors, x) + self.mean

    def reconstruction(
            self,
            x: np.ndarray,
            n_factors: int) -> np.ndarray:
        """Approximately reconstruction of x by n_factors-PCA."""
        y = self.encoding(x)
        y = self.dimension_reduction(y, n_factors)
        return self.decoding(y)

    def principal_component(self, n: int) -> np.ndarray:
        """Get n'th principal component."""
        return self.eigen_vectors[:, n].reshape((-1, 1)) + self.mean

    def relative_variance(self, n: int) -> float:
        """Get n'th relative variance."""
        return self.eigen_values[n] / np.sum(self.eigen_values)
