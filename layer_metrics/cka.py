import copy

import numpy as np
from tensorflow.keras.models import Model

from layer_metrics.layer_metrics_interface import LayerMetricsInterface

np.random.seed(2)


class CKA(LayerMetricsInterface):
    def _debiased_dot_product_similarity_helper(self, xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
        return ( xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y) + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

    def feature_space_linear_cka(self, features_x, features_y, debiased=False):
        features_x = features_x - np.mean(features_x, 0, keepdims=True)
        features_y = features_y - np.mean(features_y, 0, keepdims=True)

        dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
        normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        if debiased:
            n = features_x.shape[0]
            # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
            sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
            sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
            squared_norm_x = np.sum(sum_squared_rows_x)
            squared_norm_y = np.sum(sum_squared_rows_y)

            dot_product_similarity = self._debiased_dot_product_similarity_helper(
                dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
                squared_norm_x, squared_norm_y, n)
            normalization_x = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n))
            normalization_y = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n))

        return dot_product_similarity / (normalization_x * normalization_y)

    def scores(self,  model, X_train=None, y_train=None, allowed_layers=None, n_samples=None, **metric_params):
        output = []

        if n_samples:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:  # It uses the full training data
            sub_sampling = np.arange(X_train.shape[0])

        # OBS: creates a truncated model, removing the output layer
        F = Model(model.input, model.get_layer(index=-2).output)
        # OBS: uses the truncated model and predict what will be the outputs of the new last layer
        features_F = F.predict(X_train[sub_sampling], verbose=0)

        F_line = Model(model.input, model.get_layer(index=-2).output)#TODO: Check if this is correct
       #It will probability not work for MobileNetV2 and other convolutional architectures
        for layer_idx in allowed_layers:
            # Resblock: Conv2d, Batch N., Activation, Conv2d, Batch N.
            #if isinstance(model.get_layer(index=layer_idx + self.layer_offset), BatchNormalization):
            #_layer = model.get_layer(index=layer_idx - 1)
            _layer = F_line.get_layer(index=layer_idx - 1)
            _w = _layer.get_weights()
            _w_original = copy.deepcopy(_w)

            for i in range(0, len(_w)):
                _w[i] = np.zeros(_w[i].shape)

            _layer.set_weights(_w)
            #F_line = Model(model.input, model.get_layer(index=-2).output)
            features_line = F_line.predict(X_train[sub_sampling], verbose=0)

            _layer.set_weights(_w_original)

            score = self.feature_space_linear_cka(features_F, features_line)
            output.append((layer_idx, 1 - score))

        return output
