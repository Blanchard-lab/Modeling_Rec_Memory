""" Utility Functions For Use by HyperoptSearch """

import numpy as np

from sklearn.metrics import make_scorer, f1_score, cohen_kappa_score, accuracy_score

def custom_cross_validate(y, y_pred, groups, scoring):
    """
    Computes per-group performance metrics for predicted and actual values.

    Args:
        y (array-like): True target values.
        y_pred (array-like): Predicted target values.
        groups (array-like): Group labels (e.g., subject IDs) corresponding to each sample.
        scoring (dict): Dictionary mapping metric names to scoring functions 
            Example:
                {
                    'kappa': lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred),
                    ...
                }

    Returns:
        dict: A dictionary containing:
            - 'group_id': List of unique group identifiers.
            - 'actual': List of true labels for each group.
            - 'pred': List of predicted labels for each group.
            - <metric>: List of scores for each metric per group.
    """
    scores_by_group = {
        'group_id': [],
        'actual': [],
        'pred': []
    }
    for metric in scoring.keys():
        scores_by_group[metric] = []

    for group_id in np.unique(groups):
        scores_by_group['group_id'].append(group_id)
        mask = (groups == group_id)
        
        y_this_group = y[mask]
        y_pred_this_group = y_pred[mask]
        
        for metric, scorer in scoring.items():
            scores_by_group[metric].append(scorer(y_this_group, y_pred_this_group))
        
        scores_by_group['actual'].append(y_this_group)
        scores_by_group['pred'].append(y_pred_this_group)
    return scores_by_group