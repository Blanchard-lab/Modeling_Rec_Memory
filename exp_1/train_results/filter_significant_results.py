"""
Script to find statistically significant top-performing models across hyperopt searches.

This script:
1. Recursively searches for full_search.csv files in feature subset directories
2. Extracts the top model from each search (by avg_kappa)
3. Tests if avg_kappa is statistically significantly > 0 (alpha=0.05)
4. Returns top 5 statistically significant models with detailed metrics
5. Tracks feature subset, buffer size, and window size for each result

Usage:
    python filter_significant_results.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path

# Configuration
BASE_DIR = "/home/exx/caleb/Modeling_Rec_Memory/exp_1/train_results/exp1_feature_scaling_class_weights"
N_FOLDS = 25  # Number of participants (LOGO CV)
ALPHA = 0.05
OUTPUT_CSV = "top_5_statistically_significant_models.csv"
OUTPUT_LOG = "top_5_statistically_significant_models.txt"


def calculate_confidence_interval(avg_kappa, std_kappa, n_folds, alpha=0.05):
    """
    Calculate confidence interval for kappa score.

    Args:
        avg_kappa: Mean kappa across folds
        std_kappa: Standard deviation of kappa across folds
        n_folds: Number of cross-validation folds
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        dict with CI bounds, t-statistic, p-value, and significance flag
    """
    # Standard error
    se = std_kappa / np.sqrt(n_folds)

    # Degrees of freedom
    df = n_folds - 1

    # t-critical value for two-tailed test
    t_critical = stats.t.ppf(1 - alpha/2, df)

    # Confidence interval
    ci_lower = avg_kappa - (t_critical * se)
    ci_upper = avg_kappa + (t_critical * se)

    # t-statistic for testing H0: kappa = 0
    t_stat = avg_kappa / se if se > 0 else np.inf

    # p-value (one-tailed test: kappa > 0)
    p_value = 1 - stats.t.cdf(t_stat, df)

    # Check if significantly > 0
    is_significant = p_value < alpha

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'standard_error': se
    }


def calculate_coefficient_of_variation(avg_kappa, std_kappa):
    """Calculate coefficient of variation (CV) for stability assessment."""
    if avg_kappa == 0:
        return np.inf
    return abs(std_kappa / avg_kappa)


def parse_folder_structure(csv_path):
    """
    Extract feature subset, buffer size, and window size from path.

    Args:
        csv_path: Path to full_search.csv file

    Returns:
        dict with feature_subset, buffer_size, window_size
    """
    parts = Path(csv_path).parts

    # Find the indices for the relevant parts
    try:
        # Window/buffer folder (e.g., "0ms_buff_1sec_window")
        buffer_window_folder = parts[-2]

        # Feature subset folder (e.g., "blink_features")
        feature_subset = parts[-3]

        # Parse buffer and window from folder name
        # Format: "{buffer}ms_buff_{window}sec_window"
        folder_parts = buffer_window_folder.split('_')
        buffer_size = folder_parts[0]  # e.g., "0ms"
        window_size = folder_parts[2]  # e.g., "1sec"

        return {
            'feature_subset': feature_subset,
            'buffer_size': buffer_size,
            'window_size': window_size,
            'config': f"{feature_subset}_{buffer_window_folder}"
        }
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse folder structure from {csv_path}: {e}")
        return {
            'feature_subset': 'unknown',
            'buffer_size': 'unknown',
            'window_size': 'unknown',
            'config': 'unknown'
        }


def load_and_process_csv(csv_path):
    """
    Load a full_search.csv and extract top model with statistics.

    Args:
        csv_path: Path to CSV file

    Returns:
        dict with model info and statistical test results, or None if not significant
    """
    try:
        df = pd.read_csv(csv_path)

        # Get top model by avg_kappa
        top_model = df.iloc[0]

        # Extract metadata from path
        metadata = parse_folder_structure(csv_path)

        # Calculate statistical significance
        sig_results = calculate_confidence_interval(
            top_model['avg_kappa'],
            top_model['std_kappa'],
            N_FOLDS,
            ALPHA
        )

        # Calculate coefficient of variation
        cv = calculate_coefficient_of_variation(
            top_model['avg_kappa'],
            top_model['std_kappa']
        )

        # Only return if statistically significant
        if not sig_results['is_significant']:
            return None

        # Compile results
        result = {
            'feature_subset': metadata['feature_subset'],
            'buffer_size': metadata['buffer_size'],
            'window_size': metadata['window_size'],
            'configuration': metadata['config'],
            'trial_id': top_model['trial_id'],
            'model_name': top_model['model_name'],
            'model_params': top_model['params'],

            # Primary metrics
            'avg_kappa': top_model['avg_kappa'],
            'std_kappa': top_model['std_kappa'],
            'cv_kappa': cv,

            # Statistical significance
            'ci_95_lower': sig_results['ci_lower'],
            'ci_95_upper': sig_results['ci_upper'],
            't_statistic': sig_results['t_statistic'],
            'p_value': sig_results['p_value'],
            'standard_error': sig_results['standard_error'],

            # Overall performance metrics
            'kappa_overall': top_model['kappa_overall'],
            'avg_f1': top_model['avg_f1'],
            'std_f1': top_model['std_f1'],
            'f1_overall': top_model['f1_overall'],
            'avg_accuracy': top_model['avg_accuracy'],
            'std_accuracy': top_model['std_accuracy'],
            'accuracy_overall': top_model['accuracy_overall'],
            'balanced_accuracy': top_model['balanced_accuracy'],
            'roc_avg': top_model['roc_avg'],

            # Source file
            'source_file': csv_path
        }

        return result

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def find_all_search_csvs(base_dir):
    """
    Recursively find all full_search.csv files.

    Args:
        base_dir: Root directory to search

    Returns:
        list of paths to full_search.csv files
    """
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        if 'full_search.csv' in files:
            csv_files.append(os.path.join(root, 'full_search.csv'))
    return csv_files


def main():
    """Main execution function."""

    # Print progress to console only
    print(f"Searching for full_search.csv files in: {BASE_DIR}")
    print(f"Using n_folds = {N_FOLDS}, alpha = {ALPHA}\n")

    # Find all CSV files
    csv_files = find_all_search_csvs(BASE_DIR)
    print(f"Found {len(csv_files)} full_search.csv files\n")

    if len(csv_files) == 0:
        print("No CSV files found! Check your BASE_DIR path.")
        return

    # Process each CSV and collect significant results
    significant_results = []

    for csv_path in csv_files:
        print(f"Processing: {csv_path}")
        result = load_and_process_csv(csv_path)

        if result is not None:
            significant_results.append(result)
            print(f"  ✓ Significant! avg_kappa={result['avg_kappa']:.4f}, "
                  f"p={result['p_value']:.4f}, CI=[{result['ci_95_lower']:.4f}, "
                  f"{result['ci_95_upper']:.4f}]")
        else:
            print(f"  ✗ Not significant or error")

    print(f"\n{'='*70}")
    print(f"Total statistically significant models: {len(significant_results)}")
    print(f"{'='*70}\n")

    if len(significant_results) == 0:
        print("No statistically significant models found!")
        return

    # Convert to DataFrame and sort by avg_kappa
    results_df = pd.DataFrame(significant_results)
    results_df = results_df.sort_values('avg_kappa', ascending=False)

    # Get top 5
    top_5 = results_df.head(5)

    # Save to CSV
    csv_output_path = os.path.join(BASE_DIR, OUTPUT_CSV)
    top_5.to_csv(csv_output_path, index=False)

    print(f"Top 5 models saved to: {csv_output_path}\n")

    # Now write summary to log file (only the summary part)
    log_path = os.path.join(BASE_DIR, OUTPUT_LOG)
    with open(log_path, 'w') as log_file:
        log_file.write("="*70 + "\n")
        log_file.write(f"Total statistically significant models: {len(significant_results)}\n")
        log_file.write("="*70 + "\n\n")

        log_file.write(f"Top 5 models saved to: {csv_output_path}\n\n")

        log_file.write("="*70 + "\n")
        log_file.write("TOP 5 STATISTICALLY SIGNIFICANT MODELS\n")
        log_file.write("="*70 + "\n")

        for idx, row in top_5.iterrows():
            rank = list(top_5.index).index(idx) + 1
            log_file.write(f"\nRank {rank}:\n")
            log_file.write(f"  Feature Subset: {row['feature_subset']}\n")
            log_file.write(f"  Buffer: {row['buffer_size']}, Window: {row['window_size']}\n")
            log_file.write(f"  Model: {row['model_name']}\n")
            log_file.write(f"  Avg Kappa: {row['avg_kappa']:.4f} ± {row['std_kappa']:.4f}\n")
            log_file.write(f"  95% CI: [{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}]\n")
            log_file.write(f"  p-value: {row['p_value']:.6f}\n")
            log_file.write(f"  CV (stability): {row['cv_kappa']:.4f}\n")
            log_file.write(f"  Kappa Overall: {row['kappa_overall']:.4f}\n")
            log_file.write(f"  Balanced Accuracy: {row['balanced_accuracy']:.4f}\n")
            log_file.write(f"  F1 Overall: {row['f1_overall']:.4f}\n")

        log_file.write("\n" + "="*70 + "\n")
        log_file.write(f"CSV results saved to: {csv_output_path}\n")
        log_file.write(f"Log file saved to: {log_path}\n")
        log_file.write("="*70 + "\n")

    # Print summary to console as well
    print("="*70)
    print("TOP 5 STATISTICALLY SIGNIFICANT MODELS")
    print("="*70)

    for idx, row in top_5.iterrows():
        print(f"\nRank {list(top_5.index).index(idx) + 1}:")
        print(f"  Feature Subset: {row['feature_subset']}")
        print(f"  Buffer: {row['buffer_size']}, Window: {row['window_size']}")
        print(f"  Model: {row['model_name']}")
        print(f"  Avg Kappa: {row['avg_kappa']:.4f} ± {row['std_kappa']:.4f}")
        print(f"  95% CI: [{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}]")
        print(f"  p-value: {row['p_value']:.6f}")
        print(f"  CV (stability): {row['cv_kappa']:.4f}")
        print(f"  Kappa Overall: {row['kappa_overall']:.4f}")
        print(f"  Balanced Accuracy: {row['balanced_accuracy']:.4f}")
        print(f"  F1 Overall: {row['f1_overall']:.4f}")

    print("\n" + "="*70)
    print(f"CSV results saved to: {csv_output_path}")
    print(f"Log file saved to: {log_path}")
    print("="*70)


if __name__ == "__main__":
    main()
