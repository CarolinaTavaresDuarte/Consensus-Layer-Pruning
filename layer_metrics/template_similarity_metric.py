import copy

import numpy as np
from tensorflow.keras.models import Model

from layer_metrics.layer_metrics_interface import LayerMetricsInterface


class TemplateSimilarityMetric(LayerMetricsInterface):
    @staticmethod
    def metric(X, X_line):
        euclidian = np.linalg.norm(X-X_line, axis=1)
        return np.sum(euclidian)

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=None, n_samples=None, **metric_params):
        output = []

        # n_samples = 256 Original Paper
        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
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

            score = self.metric(features_F, features_line)
            output.append((layer_idx, 1 - score))

        return output
