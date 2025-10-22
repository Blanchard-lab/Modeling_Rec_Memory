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
    """
    Extracts features, labels, and grouping from the combined dataframe.
    Uses all columns from index 4 onwards as features (assumes first 4 are metadata).

    Args:
        data_df: Combined dataframe with pos and neg data
        label: The label column to predict

    Returns:
        tuple: (X, y, groups) - features, labels, and participant groups
    """
    # Features from the dataset — all columns from index 4 onwards
    X = data_df.iloc[:, 4:]
    print("Feature Names: ", list(X.columns))
    X = X.to_numpy()

    # Labels from the dataset - what we want to train model to predict
    y = data_df[label].to_numpy().reshape(-1)

    # Group instances by participant for cross-validation
    groups = data_df['participant'].to_numpy().reshape(-1)
    print("Participants: ", np.unique(groups))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = X_scaled

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
    "experiment_name": "experiment1.7_get_participant_info",
    "buffers": ["0","250","500"],
    "windows": ["1","2","3"],
    "guide_metric": "avg_kappa",
    "label": "scene_familiarity", # What we want to predict for experiment1 folder
    "search_length": 1000,
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

            # --- Sanity check: label balance and participant distribution ---
            print("\n=== Sanity Check ===")
            print("Overall label distribution:")
            print(data_df[experiment_settings['label']].value_counts())

            print("\nLabel counts by participant:")
            print(data_df.groupby('participant')[experiment_settings['label']].value_counts())

            # Optional: check if one participant only has one label
            single_label_participants = data_df.groupby('participant')[experiment_settings['label']].nunique()
            print("\nParticipants with only one label class:")
            print(single_label_participants[single_label_participants == 1].index.tolist())
            print("=== End Sanity Check ===\n")

            data_df = data_df.groupby('participant', group_keys=False).apply(
                lambda x: x.sample(frac=1, random_state=experiment_settings['seed'])
            ).reset_index(drop=True)

            print("\n=== Checking Within-Participant Label Order ===")
            for p in data_df['participant'].unique()[:3]:
                p_data = data_df[data_df['participant'] == p]
                labels = p_data[experiment_settings['label']].values
                print(f"Participant {p}: first 20 labels = {labels[:20]}")
                print(f"  → Are labels sorted? {all(labels[i] >= labels[i+1] for i in range(len(labels)-1))}")
            print("=== End Label Order Check ===\n")


            X, y, groups = get_data_from_df(data_df, experiment_settings['label'])
            print(y[:20])

            logo = LeaveOneGroupOut()

            n_folds = len(np.unique(groups))
            print(f"🧠 Number of cross-validation folds (unique participants): {n_folds}")
            
            for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                unique_train = set(groups[train_idx])
                unique_test = set(groups[test_idx])
                assert len(unique_test) == 1, f"Fold {i}: More than one test participant!"
                assert not (unique_train & unique_test), f"Fold {i}: Data leakage detected!"
            print("✅ Leave-One-Group-Out sanity check passed — no data leakage.\n")

            # After the data leakage check, add:
            print("=== Checking Label Distribution Per Fold ===")
            logo = LeaveOneGroupOut()
            problem_folds = []

            for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                test_participant = groups[test_idx][0]
                y_test = y[test_idx]
                unique_labels = np.unique(y_test)

                print(f"Fold {i} (Participant {test_participant}): "
                    f"Test labels = {unique_labels}, "
                    f"Count: {len(y_test)}, "
                    f"Distribution: {np.bincount(y_test.astype(int))}")

                if len(unique_labels) < 2:
                    problem_folds.append((i, test_participant, unique_labels[0]))
                    print(f"  ⚠️  WARNING: Only has label {unique_labels[0]}!")

            if problem_folds:
                print(f"\n❌ CRITICAL: {len(problem_folds)} folds have only one label class:")
                for fold_num, participant, label in problem_folds:
                    print(f"   Fold {fold_num} (Participant {participant}): only label {label}")
                print("\nThis will cause poor metrics and invalid evaluation!")
                print("Consider filtering these participants or using stratified CV.\n")
            else:
                print("✅ All folds have both label classes!\n")

            for p in data_df['participant'].unique()[:3]:
                p_data = data_df[data_df['participant'] == p]
                print(f"Participant {p}: labels = {p_data[experiment_settings['label']].values}")


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
