# interface_module.py

from abc import ABC, abstractmethod


class LayerMetricsInterface(ABC):
    @abstractmethod
    def scores(self, model, X_train=None, y_train=None, allowed_layers=None, n_samples=None, **metric_params):
        pass
