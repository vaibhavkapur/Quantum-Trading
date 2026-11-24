import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Reducer:
    """Fit on training only, then freeze. All classifiers see these same angles."""
    def __init__(self, components: int):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=components, svd_solver="full")

    def fit(self, x):
        z = self.pca.fit_transform(self.scaler.fit_transform(x))
        self.scale = np.maximum(z.std(0), 1e-8)
        return self

    def transform(self, x):
        z = self.pca.transform(self.scaler.transform(x)) / self.scale
        return np.tanh(z / 2) * np.pi
