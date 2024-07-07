import copy
from typing import Tuple

import numpy as np
import numpy.typing as npt
from tensorflow.keras.models import Model

from layer_metrics.layer_metrics_interface import LayerMetricsInterface
from netrep.metrics.stochastic import GaussianStochasticMetric


class GaussianStochastic(LayerMetricsInterface):
    @staticmethod
    def get_means_and_covs(
            X: npt.NDArray[np.float64],
            y: [npt.NDArray[np.int64]],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Helper method that computes class-conditional means and covariances."""

        classes = np.unique(y)
        K = len(classes)
        means = np.stack([np.mean(X[y == k], 0) for k in range(K)], 0)
        covs = np.stack([np.cov(X[y == k].T) for k in range(K)], 0)
        return means, covs

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=None, n_samples=None, **metric_params):
        alpha = metric_params['alpha']
        if alpha is None:
            alpha = 0
        output = []

        # Convert the one-hot encoding to an integer encoding
        y_train = np.argmax(y_train, axis=1)
        # n_samples = X
        if n_samples:
            sub_sampling = [np.random.choice(np.where(y_train == value)[0], n_samples, replace=False) for value in
                            np.unique(y_train)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        F = Model(model.input, model.get_layer(index=-2).output)
        features_F = F.predict(X_train[sub_sampling], verbose=0)

        F_line = Model(model.input, model.get_layer(index=-2).output)

        for layer_idx in allowed_layers:
            _layer = F_line.get_layer(index=layer_idx - 1)
            _w = _layer.get_weights()
            _w_original = copy.deepcopy(_w)

            for i in range(0, len(_w)):
                _w[i] = np.zeros(_w[i].shape)

            _layer.set_weights(_w)
            features_line = F_line.predict(X_train[sub_sampling], verbose=0)

            _layer.set_weights(_w_original)

            # compute the class-conditional means and covariances for each network
            means_x1, covs_x1 = self.get_means_and_covs(features_F, y_train)
            means_x2, covs_x2 = self.get_means_and_covs(features_line, y_train)

            # compile into mu and sigma dicts for easier processing
            means = (means_x1, means_x2)
            covs = (covs_x1, covs_x2)

            Xi = (means[0], covs[0])
            Xj = (means[1], covs[1])

            metric = GaussianStochasticMetric(alpha=alpha)
            metric.fit(Xi, Xj)
            dist = metric.score(Xi, Xj)

            output.append((layer_idx, dist))

        return output
