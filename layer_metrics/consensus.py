from collections import defaultdict
import gc

from layer_metrics.layer_metrics_modules import layer_metrics_modules
from layer_metrics.layer_metrics_interface import LayerMetricsInterface
from layer_metrics import load_layer_metric_class

#metrics_list = [
#    'cka',
#    'gaussian_stochastic_0',
#    'gaussian_stochastic_1',
#    'gaussian_stochastic_2',
#    'permutation',
#    'linear_0',
#    'linear_1',
#    'wasserstein_distance',
#]

class Consensus(LayerMetricsInterface):
    def scores(self,  model, metrics_list, X_train=None, y_train=None, allowed_layers=None, n_samples=None, **metric_params):
        # Computing scores for each metric
        scores = []
        for metric in metrics_list:
            alpha = None
            if metric.endswith('_0') or metric.endswith('_1') or metric.endswith('_2'):
                # Extract the number from the end of the string
                alpha = int(metric[-1])
                # Remove the trailing '_0' or '_1' from the original string
                metric = metric[:-2]

            layer_method = load_layer_metric_class(metric)
            score = layer_method.scores(model, X_train, y_train, allowed_layers, n_samples, alpha=alpha)
            print(f"Metric {metric} score: {score}")
            del layer_method
            gc.collect()
            scores.append(score)

        metrics_scores_list_sorted = []

        # For each metric, sort the scores and index the blocks with a ranking
        for sublist in scores:
            # Sort the sublist based on the second element of the tuple (index 1)
            sorted_sublist = sorted(sublist, key=lambda x: x[1])

            # Assign ranks using enumerate starting from 1
            ranked_sublist = [(index + 1, item[0], item[1]) for index, item in enumerate(sorted_sublist)]

            # Append the ranked sublist to the new list
            metrics_scores_list_sorted.append(ranked_sublist)

        # Create a defaultdict to store cumulative rankings for each identifier
        cumulative_rankings = defaultdict(float)

        # Iterate through the nested lists and update cumulative rankings
        for sublist in metrics_scores_list_sorted:
            for ranking, identifier, score in sublist:
                cumulative_rankings[identifier] += ranking

        # Convert the defaultdict to a list of tuples (identifier, cumulative ranking)
        result = [(identifier, cumulative_ranking) for identifier, cumulative_ranking in cumulative_rankings.items()]

        # Sort the result based on cumulative rankings
        result.sort(key=lambda x: x[1])

        return result
