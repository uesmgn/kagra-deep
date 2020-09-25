from sklearn.manifold import TSNE

__all__ = [
    'TSNE'
]


class TSNE:
    def __init__(self, n_components=2):
        self.tsne = TSNE(n_components=n_components)

    def fit_transform(self, data):
        data = self.tsne.fit_transform(data)
        return data
