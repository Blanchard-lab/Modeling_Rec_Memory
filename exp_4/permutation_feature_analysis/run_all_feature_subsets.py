import subprocess
import os

# Paste your absolute path to train.py here:
TRAIN_SCRIPT_PATH = "/home/exx/caleb/familiarity/experiment4/permutation_feature_analysis/training/train.py"
FEATURE_DIRS_FILE = "feature_dirs.txt"

def main():
    if not os.path.isfile(FEATURE_DIRS_FILE):
        print(f"ERROR: Could not find {FEATURE_DIRS_FILE} in the current directory.")
        return

    with open(FEATURE_DIRS_FILE, "r") as f:
        feature_dirs = [line.strip() for line in f if line.strip()]

    if not feature_dirs:
        print(f"ERROR: {FEATURE_DIRS_FILE} is empty or only contains blank lines.")
        return

    print(f"Using train.py at: {TRAIN_SCRIPT_PATH}")
    print(f"Found {len(feature_dirs)} feature subset folder(s) in {FEATURE_DIRS_FILE}")

    for feature_dir in feature_dirs:
        print(f"\n=== Running train.py for feature subset: {feature_dir} ===\n")
        result = subprocess.run(
            ["python", TRAIN_SCRIPT_PATH, "--feature_dir", feature_dir]
        )
        if result.returncode != 0:
            print(f"ERROR: train.py failed for {feature_dir} (return code {result.returncode}).")
            # Optional: break here or continue to next if you want to skip failures
            break

if __name__ == "__main__":
    main()
