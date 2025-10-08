"""
This module performs hyperparameter optimization for the familiarity detection
machine learning models.

Utilizes the HyperoptSearch wrapper on the hyperopt library to perform the
hyperparameter search. Includes functions to prepare the dataset,
define the search space for hyperparameters, and save the results.

Usage:
    1. Set experiment settings in the `__main__` block.
        - `experiment_name`: A string for naming the experiment (used in file
          paths).
        - `buffers`: List of buffer sizes to use in the experiment (e.g., ["0", "250", "500"]).
        - `windows`: List of window sizes (e.g., ["1", "2", "3"]).
        - `guide_metric`: The metric to guide the optimization (e.g., 'avg_kappa').
        - `label`: The label column to predict (e.g., 'study_status').
        - `search_length`: Number of iterations for hyperparameter search (e.g., 500).
        - `seed`: Random seed for reproducibility.

    2. Additonal notes for running the experiment:
        - Modify the `buffers` and `windows` to match the buffer sizes and window
          sizes you wish to explore. Note that these combinations must exist as extracted
          features files (e.g. 'pos_pytrack_buff{bufffer_size}ms_{window_size}_sec.csv)

    3. Once the settings are adjusted, run the script to begin hyperparameter optimization.
"""

from hyperopt_search import HyperoptSearch
import numpy as np
import pandas as pd
import os
import csv
import json
from hyperopt import hp, tpe
from sklearn.model_selection import LeaveOneGroupOut
import argparse
parser = argparse.ArgumentParser(description="Run Experiment 1 with a specified feature subset directory.")
parser.add_argument("--feature_dir", type=str, required=True, help="Path to the folder containing feature subset CSVs.")
args = parser.parse_args()
feature_dir = args.feature_dir


def get_param_space(num_samples):
    """
    Defines the hyperopt search space for chosen hyperparameters
    of various chosen machine learning classifiers.

    Args:
        num_samples (int): The number of samples in the dataset, used to
        determine the upper bound for certain hyperparameters.

    Returns:
        dict: A dictionary matching the format of the param_space constructor
        parameter in the HyperoptSearch class (see hyperopt_search module).
    """
    space = hp.choice('classifier_type', [
        {
            'type': 'AdaBoost',
            'n_estimators': hp.choice('AdaBoost_n_estimators', list(range(30, 100, 11))),
            'learning_rate': hp.choice('AdaBoost_learning_rate', list(np.linspace(0.1, 4.1, 50))),
            'algorithm': hp.choice('AdaBoost_algorithm', ['SAMME', 'SAMME.R'])
        },
        {
            'type': 'naive_bayes',
        },
        {
            'type': 'logistic_regression',
            'penalty' : hp.choice('logistic_regression_penalty', ['l2', None]),
            'solver' : hp.choice('logistic_regression_solver', ['lbfgs', 'newton-cg', 'sag', 'saga'])
        },
        {
            'type': 'SVC',
            'C': hp.choice('C', [0.1, 1, 10, 100, 1000]),
            'gamma': hp.choice('gamma', [1, 0.1, 0.01, 0.001, 0.0001]),
            'kernel': hp.choice('kernal', ['rbf', 'sigmoid']),

        },
        {
            'type': 'randomforest',
            'n_estimators': hp.choice('n_estimators', list(range(25, 600, 2))),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'max_features': hp.choice('max_features', [None, "sqrt", "log2"]),

        },
        {
            'type': 'knn',
            # only have num_samples samples (less after split), limits upper bound of number neighbors
            #'n_neighbors': hp.choice('knn_n_neighbors', list(range(25, num_samples - 60, 5)))
            'n_neighbors': hp.choice('knn_n_neighbors', list(range(1, min(50, max(2, num_samples // 2)), 2)))

        }
    ])
    return space


def process(df):
    """
    Were not filtering out any instances for this experiment

    Returns:
        pandas.DataFrame: The filtered dataset containing only relevant
        instances.
    """
    # Example use below:
    # df = df[(df['recall_status'] == 0) | (df['recall_status']== 0.5)] This line was used for the recall failure experiment
    return df


def get_data_from_df(data_df, label):
    # Only keep validated features
    CORE_FEATURES = [
        "pupil_AUC",
        "pupil_slope",
        "avg_pupil_size_downsample",
        "pupil_mean",
        "avg_pupil_size",
        "peak_blink_duration",
        "blink_cnt"
    ]

    # Ensure all selected features are present in the data
    available_features = [f for f in CORE_FEATURES if f in data_df.columns]
    missing = [f for f in CORE_FEATURES if f not in data_df.columns]
    if missing:
        print(f"⚠️ Warning: The following expected features are missing from the dataset: {missing}")

    X = data_df[available_features]
    print("Using Selected Features:", list(X.columns))
    X = X.to_numpy()

    y = data_df[label].to_numpy().reshape(-1)
    groups = data_df['participant'].to_numpy().reshape(-1)
    return X, y, groups


def write_search_results(results_dir, search_results_df):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, "full_search.csv")
    search_results_df.to_csv(results_path, index=False)
    print(f"Search Results written to {results_path}")


def write_trial_results(results_dir, results_by_trial):
    trial_results_dir = os.path.join(results_dir, "trials")
    if not os.path.exists(trial_results_dir):
        os.makedirs(trial_results_dir)
    for trial_results in results_by_trial:
        filename = f"trial_id_{trial_results['trial_id']}.csv"
        filepath = os.path.join(trial_results_dir, filename)
        trial_results['results_by_group'].to_csv(filepath, index=False)


def write_settings_file(experiment_dir, experiment_settings):
    """Write Experiment Settings to a JSON File"""
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        print(f"Directory {experiment_dir} created.")
    else:
        print(f"\nERROR: \nExperiment {experiment_settings['experiment_name']} already exists,",
              "please choose a new experiment name or\ndelete the ",
              f"directory {experiment_dir}\n")
        exit()

    settings_filename = os.path.join(experiment_dir, "experiment_settings.json")
    with open(settings_filename, "w") as file:
        json.dump(experiment_settings, file, indent=4)



if __name__ == "__main__":
    # Set your settings here
    experiment_settings = {
    "experiment_name": "experiment1.7StatisticalyRelevantFeaturesOnly", # Name of the experiment
    "buffers": ["500"],
    "windows": ["3"],
    "guide_metric": "avg_kappa",
    "label": "scene_familiarity", # What we want to predict for experiment1 folder
    "search_length": 750,
    "seed": 3789,
    "experiment_description": "Train classifier to detect scene_familiarity (0 vs 1) using both pos and neg instances."
    }

    #experiment_dir = os.path.join(
    #    "../train_results", experiment_settings['experiment_name']
    #    )
    experiment_dir = os.path.join(
        "../train_results", experiment_settings['experiment_name'], os.path.basename(feature_dir)
        )

    write_settings_file(experiment_dir, experiment_settings)

    for buffer_size in experiment_settings['buffers']:
        for window_size in experiment_settings['windows']:
            results_dir = os.path.join(experiment_dir,
                                       f"{buffer_size}ms_buff_{window_size}sec_window"
                                       )

            #base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Features"))

            pos_file = os.path.join(feature_dir, f'pos_pytrack_buff_{buffer_size}ms_{window_size}_sec.csv')
            neg_file = os.path.join(feature_dir, f'neg_pytrack_{window_size}_sec_window.csv')



            pos_df = pd.read_csv(pos_file)
            neg_df = pd.read_csv(neg_file)

            data_df = pd.concat([pos_df, neg_df], ignore_index=True)
            data_df = process(data_df)

            X, y, groups = get_data_from_df(data_df, experiment_settings['label'])


            '''
            # Prepare Data
            data_file = (
                f'../Features/pos_pytrack_buff_{buffer_size}ms_{window_size}_sec.csv'
            )
            X, y, groups = get_data(data_file, experiment_settings['label'])
            '''
            # Set up parameter space for optimization
            num_samples = X.shape[0]
            space = get_param_space(num_samples)

            optimizer = HyperoptSearch(
                X, y,
                cross_val=LeaveOneGroupOut(),
                groups=groups,
                param_space=space,
                guide_metric=experiment_settings['guide_metric'],
                seed=experiment_settings['seed']
            )
            optimizer.begin_search(
                algorithm=tpe.suggest,
                search_length=experiment_settings['search_length']
            )

            search_results_df = optimizer.get_search_results()
            write_search_results(results_dir, search_results_df)

            results_by_trial = optimizer.get_grouped_results_by_trial()
            write_trial_results(results_dir, results_by_trial)
