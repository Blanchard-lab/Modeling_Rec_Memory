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
parser = argparse.ArgumentParser(description="Run Experiment 4 with a specified feature subset directory.")
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
            'kernel': hp.choice('kernel', ['rbf', 'sigmoid']),

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
            'n_neighbors': hp.choice('knn_n_neighbors', list(range(25, num_samples - 60, 5)))
        }
    ])
    return space


def process(df):
    """
    Preprocesses the input dataset by filtering to include only
    instances of recall failure. Adjust as necessary per your experiment needs.

    Returns:
        pandas.DataFrame: The filtered dataset containing only relevant
        instances.
    """
    df = df[(df['recall_status'] == 0) | (df['recall_status']== 0.5)]
    return df

def balance_classes(df, label):
    """
    Downsamples each class to the size of the smallest class to balance the dataset.
    Args:
        df (pd.DataFrame): The filtered DataFrame (e.g., after process()).
        label (str): Name of the label column to balance on.
    Returns:
        pd.DataFrame: Balanced DataFrame with equal samples from each class.
    """
    class_counts = df[label].value_counts()
    min_count = class_counts.min()
    balanced_df = (
        df.groupby(label, group_keys=False)
          .apply(lambda x: x.sample(min_count, random_state=42))
          .reset_index(drop=True)
    )
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Participants before balancing: {df['participant'].nunique()}")
    print(f"Participants after balancing: {balanced_df['participant'].nunique()}")

    
    print("\nClass distribution by participant after balancing:")
    participant_class_dist = balanced_df.groupby(['participant', label]).size().unstack(fill_value=0)
    print(participant_class_dist)
    print(f"\nParticipants with imbalanced classes: {(participant_class_dist[0.0] != participant_class_dist[1.0]).sum()}")

    return balanced_df

def get_data(filename, label):
    """
    Reads and processes the dataset, extracting features, labels, and grouping
    by participants for cross-validation.

    Args:
        filename (str): The path to the CSV file containing the features dataset.
        label (str): The name of the column in the features dataset to be predicted.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The feature matrix extracted from the dataset.
            - y (numpy.ndarray): The label vector extracted from the dataset.
            - groups (numpy.ndarray): The group identifiers for cross-validation.
    """
    data_df = pd.read_csv(filename)
    data_df = process(data_df) # exclude unwanted instances
    data_df = balance_classes(data_df, label)

    # print(f"\nFile: {os.path.basename(filename)}")
    # print("Label distribution after filtering and balancing:")
    # print(data_df[label].value_counts())
    # print("Total instances:", len(data_df))
    # print("------")

    # Features from the dataset — all eye gaze features
    X = data_df.iloc[:, 4:]
    print("Feature Names: ", X.columns)
    X = X.to_numpy()

    # Labels from the dataset - what we want to train model to predict
    y = pd.DataFrame(data_df[label], columns=[label])
    y = y.to_numpy()
    y = y.reshape(-1)

    # Group instances by participant for cross-validation
    groups = pd.DataFrame(data_df['participant'], columns=['participant'])
    groups = groups.to_numpy()
    groups = groups.reshape(-1)
    print("Participants: ", np.unique(groups))
    print("=== PRE-BALANCE DIAGNOSTICS ===")
    print(f"Total samples: {len(data_df)}")
    print(f"Total participants: {data_df['participant'].nunique()}")
    print(f"Class distribution:\n{data_df[label].value_counts()}")
    print(f"Samples per participant:\n{data_df.groupby('participant').size().describe()}")

    return X, y, groups
    #return None, None, None


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
    "experiment_name": "experiment5PerFeA",
    "buffers": ["0","250","500"],
    "windows": ["1","2","3"],
    "guide_metric": "avg_kappa",
    "label": "study_status",
    "search_length": 1000,
    "seed": 3789,
    "experiment_description": "Investigates whether scenes that elicited a familiarity response but resulted in failed or partial recall still show encoding traces. This tests if study status can be recovered even when memory retrieval fails."
    }

    experiment_dir = os.path.join(
        "../train_results", experiment_settings['experiment_name'], os.path.basename(feature_dir)
        )
    write_settings_file(experiment_dir, experiment_settings)

    for buffer_size in experiment_settings['buffers']:
        for window_size in experiment_settings['windows']:
            results_dir = os.path.join(experiment_dir,
                                       f"{buffer_size}ms_buff_{window_size}sec_window"
                                       )
            # Prepare Data
            #base_dir = "/home/exx/caleb/familiarity/feature-gen0/Features"
            data_file = os.path.join(feature_dir, f'pos_pytrack_buff_{buffer_size}ms_{window_size}_sec.csv')
            X, y, groups = get_data(data_file, experiment_settings['label'])

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
