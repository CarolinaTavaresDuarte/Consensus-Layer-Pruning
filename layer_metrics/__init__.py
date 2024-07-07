# layer_metrics/__init__.py
from layer_metrics.layer_metrics_modules import layer_metrics_modules


def load_layer_metric_class(metric_name):
    metric_info_dict = layer_metrics_modules[metric_name.lower()]
    class_name = metric_info_dict['class_name']
    module_name = metric_info_dict['module_name']

    # Dynamically import the module and retrieve the class
    module = __import__(f"layer_metrics.{module_name}", fromlist=[class_name])
    score_class = getattr(module, class_name)

    # Instantiate the class
    return score_class()
