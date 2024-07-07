# Note: all metric names (the keys from the main dictionary) must be written
# in lower case. E.g: write 'cka' instead of 'CKA' or 'Cka'.
# Note 2: please keep all the dictionary keys in alphabetical order

layer_metrics_modules = {
    'cka': {
        'class_name': 'CKA',
        'module_name': 'cka'
    },
    'consensus': {
        'class_name': 'Consensus',
        'module_name': 'consensus'
    },
    'gaussian_stochastic': {
        'class_name': 'GaussianStochastic',
        'module_name': 'gaussian_stochastic'
    },
    'linear': {
        'class_name': 'Linear',
        'module_name': 'linear'
    },
    'permutation': {
        'class_name': 'Permutation',
        'module_name': 'permutation'
    },
    'template_similarity_metric': {
        'class_name': 'TemplateSimilarityMetric',
        'module_name': 'template_similarity_metric'
    },
    'wasserstein_distance': {
        'class_name': 'WassersteinDistance',
        'module_name': 'wasserstein_distance'
    },
}
