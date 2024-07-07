import numpy as np
import scipy.stats
from tensorflow.keras.models import Model
import copy

from layer_metrics.layer_metrics_interface import LayerMetricsInterface

class WassersteinDistance(LayerMetricsInterface):

    """Class for calculating the Wasserstein distance between activation distributions."""
    
    def __init__(self):
        pass

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=None, n_samples=None, **metric_params):
        output = []

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

            # Calculate the Wasserstein distance between activation distributions
            wasserstein_dist = scipy.stats.wasserstein_distance(features_F.flatten(), features_line.flatten())

            output.append((layer_idx, wasserstein_dist))

        return output
