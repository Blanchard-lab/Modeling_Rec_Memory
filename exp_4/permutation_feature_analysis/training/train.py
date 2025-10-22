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
from sklearn.preprocessing import StandardScaler
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
            'n_neighbors': hp.choice('knn_n_neighbors', list(range(25, num_samples - 60, 5)))
        }
    ])
    return space


def process(df):
    """
    Filters the dataset to include only studied scenes (study_status == 1)
    and recall_status in [0, 0.5, 1]. Then maps 0.5 to 0 to represent recall failure.

    Returns:
        pandas.DataFrame: The filtered and updated dataset.
    """
    df = df[
        (df["study_status"] == 1) &
        ((df["recall_status"] == 0) | (df["recall_status"] == 0.5) | (df["recall_status"] == 1))
    ]

    # Map recall_status = 0.5 to 0 to represent failure
    df["recall_status"] = df["recall_status"].apply(lambda x: 0 if x == 0.5 else x)

    return df


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
    print(f"Total samples: {len(data_df)}")
    print(f"Participants: {data_df['participant'].nunique()}")
    data_df = process(data_df) # exclude unwanted instances
    print(f"\n=== AFTER FILTERING (scene_familiarity==1) ===")
    print(f"Total samples: {len(data_df)}")
    print(f"Participants: {data_df['participant'].nunique()}")
    samples_per_participant = data_df.groupby('participant').size()
    print(f"\nSamples per participant: min={samples_per_participant.min()}, max={samples_per_participant.max()}, mean={samples_per_participant.mean():.1f}")

    class_dist_per_participant = data_df.groupby(['participant', label]).size().unstack(fill_value=0)
    print("\n=== Per-participant class distribution ===")
    print(class_dist_per_participant)

    MIN_SAMPLES_PER_CLASS = 5  # At least 5 samples of EACH class

    # Filter participants who have at least MIN_SAMPLES_PER_CLASS of both classes
    valid_participants = class_dist_per_participant[
        (class_dist_per_participant[0.0] >= MIN_SAMPLES_PER_CLASS) &
        (class_dist_per_participant[1.0] >= MIN_SAMPLES_PER_CLASS)
    ].index

    print(f"\nParticipants with <{MIN_SAMPLES_PER_CLASS} samples of either class:")
    problematic = class_dist_per_participant[
        (class_dist_per_participant[0.0] < MIN_SAMPLES_PER_CLASS) |
        (class_dist_per_participant[1.0] < MIN_SAMPLES_PER_CLASS)
    ]
    print(problematic)

    data_df = data_df[data_df['participant'].isin(valid_participants)]

    print(f"\n=== AFTER REMOVING PARTICIPANTS WITH <{MIN_SAMPLES_PER_CLASS} PER CLASS ===")
    print(f"Total samples: {len(data_df)}")
    print(f"Participants: {data_df['participant'].nunique()}")
    print(f"Samples per participant: min={data_df.groupby('participant').size().min()}, max={data_df.groupby('participant').size().max()}")

    if len(data_df) == 0:
        raise ValueError(f"No participants have >={MIN_SAMPLES_PER_CLASS} samples per class!")



    data_df = (data_df.groupby('participant', group_keys=False)
                      .apply(lambda x: x.sample(frac=1, random_state=42))
                      .reset_index(drop=True))



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
    print(f"Total samples: {len(data_df)}")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")

    print("\n=== FEATURE ANALYSIS (BEFORE SCALING) ===")
    feature_names = data_df.columns[4:].tolist()
    feature_stats = pd.DataFrame(X, columns=feature_names).describe()
    print(feature_stats)

    print("\n=== FEATURE RANGES (BEFORE SCALING) ===")
    feature_ranges = X.max(axis=0) - X.min(axis=0)
    for i, name in enumerate(feature_names):
        print(f"{name}: {feature_ranges[i]:.3f}")

    print("\n=== FEATURE CORRELATIONS WITH TARGET ===")
    temp_df = pd.DataFrame(X, columns=feature_names)
    temp_df['target'] = y
    correlations = temp_df.corr()['target'].drop('target')
    print("Correlations (sorted by absolute value):")
    print(correlations.sort_values(key=abs, ascending=False))

    # ADD FEATURE SCALING HERE
    print("\n=== APPLYING FEATURE SCALING ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== FEATURE ANALYSIS (AFTER SCALING) ===")
    print(f"All features now have mean ≈ 0 and std ≈ 1")
    scaled_stats = pd.DataFrame(X_scaled, columns=feature_names).describe()
    print("Mean values:", scaled_stats.loc['mean'].round(3).values)
    print("Std values:", scaled_stats.loc['std'].round(3).values)

    return X_scaled, y, groups
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
    "experiment_name": "experiment4ClassWeightsFeatureScaling",
    "buffers": ["0","250","500"],
    "windows": ["1","2","3"],
    "guide_metric": "avg_kappa",
    "label": "recall_status",
    "search_length": 1000,
    "seed": 3789,
    "experiment_description": "Experiment4"
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
            #base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Features"))
            data_file = os.path.join(feature_dir, f'pos_pytrack_buff_{buffer_size}ms_{window_size}_sec.csv')
            X, y, groups = get_data(data_file, experiment_settings['label'])

            logo = LeaveOneGroupOut()
            for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                unique_train = set(groups[train_idx])
                unique_test = set(groups[test_idx])
                assert len(unique_test) == 1, f"Fold {i}: More than one test participant!"
                assert not (unique_train & unique_test), f"Fold {i}: Data leakage detected!"
            print("✅ Leave-One-Group-Out sanity check passed — no data leakage.\n")

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
