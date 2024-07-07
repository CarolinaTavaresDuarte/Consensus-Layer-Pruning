import copy

import numpy as np
from tensorflow.keras.models import Model

from layer_metrics.layer_metrics_interface import LayerMetricsInterface
from netrep.metrics.linear import LinearMetric


class Linear(LayerMetricsInterface):
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

            n_classes = len(np.unique(y_train))
            idx_train = np.random.choice(np.arange(n_classes), int(n_classes * .8), replace=False)

            # everything left over is test set
            idx_test = np.array(list(set(np.arange(n_classes)).difference(idx_train)))

            X1_train, X1_test = features_F[idx_train], features_F[idx_test]
            X2_train, X2_test = features_line[idx_train], features_line[idx_test]

            metric = LinearMetric(
                alpha=alpha,
                center_columns=True,
                score_method="angular",
            )
            metric.fit(X1_train, X2_train)
            dist = metric.score(X1_test, X2_test)

            output.append((layer_idx, dist))

        return output
