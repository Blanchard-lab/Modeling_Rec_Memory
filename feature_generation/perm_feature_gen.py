mport fileinput
import os
import random
import statistics
import warnings
from random import randrange

import numpy as np
import pandas as pd

from PyTrack.Stimulus import Stimulus
from PyTrack.formatBridge import generateCompatibleFormat

Ident = 0
DESCRIPTIVE_STAT_PATH = '../descriptive_stats_fixation_blink_saccade_microsaccade_features.txt'

feature_subset_name = "fixation_blink_saccade_microsaccade_features"
output_dir = f"../{feature_subset_name}"

column_types = {
    'timestamp': str,
    'type': str,
    'scene': str,
    'scene_num': object,
    'order': int,
    'djv_reported': int,
    'left_eye_pupil_diameter': float,
    'left_eye_pupil_position_x': float,
    'left_eye_pupil_position_y': float,
    'left_eye_openness': float,
    'right_eye_pupil_diameter': float,
    'right_eye_pupil_position_x': float,
    'right_eye_pupil_position_y': float,
    'right_eye_openness': float,
    'gaze_direction_x': float,
    'gaze_direction_y': float,
    'gaze_direction_z': float,
    'gaze_origin_x': float,
    'gaze_origin_y': float,
    'gaze_origin_z': float,
    'gaze_convergence_distance': float
}

def extract_instances(window_size, buffer_size):
    """Create one csv with all yoked positive and negative instances."""
    global Ident


    print(f"\nInstance extraction for {window_size} second window and {buffer_size} ms buffer:")

    processed_log_path = "/home/exx/caleb/familiarity/ProcessedLogs"

    warnings.filterwarnings("ignore", category=FutureWarning)

    yoked_pos_instances = []
    yoked_neg_instances = []

    descriptive_stats = {
        "times_to_press_button": [],
        "scene_lengths": [],
        "num_begin_with_blink": 0,
        "num_double_button_presses": 0,
        "num_button_press_too_soon": 0,
        "num_valid_pos_instances": 0,
        "num_valid_neg_instances": 0,
        "num_yoked_pairs": 0,
        "num_pos_dropped_due_to_no_neg": 0,
    }

    for file_name in os.listdir(processed_log_path):
        if not file_name.endswith('.csv'):
            continue

        file_path = os.path.join(processed_log_path, file_name)
        current_df = pd.read_csv(file_path, dtype=column_types)
        unique_scene_names = current_df['scene'].unique()


        test_scenes_no_familiarity = {
            order: rows.reset_index(drop=True)
            for order, rows in current_df[current_df['type'] == 't'].groupby('order')
            if not (rows['scene_familiarity'] == 1).any()
        }

        for scene_name in unique_scene_names:
            scene_name_rows = current_df[(current_df['scene'] == scene_name) & (current_df['type'] == 't')]
            if len(scene_name_rows) == 0:
                continue

            scene_start = scene_name_rows.index[0]
            scene_end = scene_name_rows.index[-1]
            reported_rows = scene_name_rows[scene_name_rows['djv_reported'] == 1]
            if len(reported_rows) == 0:
                continue

            reported_start = reported_rows.index[0]
            reported_end = reported_rows.index[-1]
            num_reported_rows = reported_end - reported_start
            if num_reported_rows > 100:
                descriptive_stats["num_double_button_presses"] += 1
                continue

            instance_end = int(reported_start - int(buffer_size) * 0.12)
            instance_start = int(instance_end - int(window_size) * 120)

            if instance_start < scene_start:
                descriptive_stats['num_button_press_too_soon'] += 1
                continue

            if (
                current_df.iloc[instance_start]['left_eye_pupil_diameter'] == -1
                or current_df.iloc[instance_start + 1]['left_eye_pupil_diameter'] == -1
                or current_df.iloc[instance_start]['right_eye_pupil_diameter'] == -1
                or current_df.iloc[instance_start + 1]['right_eye_pupil_diameter'] == -1
            ):
                descriptive_stats["num_begin_with_blink"] += 1
                continue

            scene_start_time = int(float(current_df.iloc[scene_start]['timestamp']))
            reported_time = int(float(current_df.iloc[reported_start]['timestamp']))
            scene_end_time = int(float(current_df.iloc[scene_end]['timestamp']))

            descriptive_stats["times_to_press_button"].append(reported_time - scene_start_time)
            descriptive_stats["scene_lengths"].append(scene_end_time - scene_start_time)


            pos_instance = current_df.iloc[instance_start:instance_end].copy()

            neg_found = False
            for order, neg_scene_rows in test_scenes_no_familiarity.items():
                max_start = len(neg_scene_rows) - (int(window_size) * 120 + 10)
                if max_start <= 0:
                    continue
                tries = 10
                while tries > 0:
                    instance_start_neg = randrange(max_start)
                    if (
                        neg_scene_rows.iloc[instance_start_neg]['left_eye_pupil_diameter'] == -1
                        or neg_scene_rows.iloc[instance_start_neg + 1]['left_eye_pupil_diameter'] == -1
                        or neg_scene_rows.iloc[instance_start_neg]['right_eye_pupil_diameter'] == -1
                        or neg_scene_rows.iloc[instance_start_neg + 1]['right_eye_pupil_diameter'] == -1
                    ):
                        tries -= 1
                        continue
                    neg_instance = neg_scene_rows.iloc[instance_start_neg:instance_start_neg + (int(window_size) * 120)].copy()
                    neg_instance['Identifier'] = Ident
                    Ident += 1
                    yoked_pos_instances.append(pos_instance)
                    yoked_neg_instances.append(neg_instance)
                    descriptive_stats["num_valid_pos_instances"] += 1
                    descriptive_stats["num_valid_neg_instances"] += 1
                    descriptive_stats["num_yoked_pairs"] += 1
                    neg_found = True
                    break
                if neg_found:
                    break
            if not neg_found:
                descriptive_stats["num_pos_dropped_due_to_no_neg"] += 1

    result_df_pos = pd.concat(yoked_pos_instances, ignore_index=True) if yoked_pos_instances else pd.DataFrame()
    result_df_neg = pd.concat(yoked_neg_instances, ignore_index=True) if yoked_neg_instances else pd.DataFrame()

    convert_participant_col_to_int(result_df_pos)
    convert_participant_col_to_int(result_df_neg)

    write_stats_to_file(window_size, buffer_size, descriptive_stats)

    result_df_pos.to_csv(os.path.join(output_dir, f'pos_file_buff_{buffer_size}ms_{window_size}_sec.csv'), index=False)
    result_df_neg.to_csv(os.path.join(output_dir, f'neg_file_{window_size}_sec_window.csv'), index=False)


def convert_participant_col_to_int(df):
    """Helper Function for extract_instances"""
    if 'participant' in df.columns:
        df['participant'] = df['participant'].astype(str).str.replace(r'\D', '', regex=True)
        df['participant'] = df['participant'].astype(int)

def write_stats_to_file(window_size, buffer_size, descriptive_stats):
    """Helper Function for extract_instances"""
    with open(DESCRIPTIVE_STAT_PATH, 'a', encoding='utf-8') as file:
        file.write(
            f"\nInstance extraction for {window_size} second window and {buffer_size} ms buffer:\n")
        file.write(
            f"\tTotal valid yoked pairs: {descriptive_stats['num_yoked_pairs']}\n")
        file.write(
            f"\tTotal valid positives: {descriptive_stats['num_valid_pos_instances']}\n")
        file.write(
            f"\tTotal valid negatives: {descriptive_stats['num_valid_neg_instances']}\n")
        file.write(
            f"\tTotal positives dropped due to no yoked negative: {descriptive_stats['num_pos_dropped_due_to_no_neg']}\n")
        file.write(
            f"\tTotal number of excluded scenes due to likely double button press: {descriptive_stats['num_double_button_presses']}\n")
        file.write(
            f"\tTotal number of excluded scenes due to button press occuring too soon: {descriptive_stats['num_button_press_too_soon']}\n")
        file.write(
            f"\tTotal number of excluded scenes due to blink at beginning of instance onset: {descriptive_stats['num_begin_with_blink']}\n")
        if descriptive_stats["times_to_press_button"]:
            file.write(
                f"\tAverage time to press button (seconds): {round(statistics.mean(descriptive_stats['times_to_press_button']) / 1000,2)}\n")
            file.write(
                f"\tStandard deviation of time to press button after scene starts (seconds): {round(statistics.stdev(descriptive_stats['times_to_press_button']) / 1000,2)}\n")
        if descriptive_stats["scene_lengths"]:
            file.write(
                f"\tAverage scene length (seconds): {round(statistics.mean(descriptive_stats['scene_lengths']) / 1000,2)}\n")


def make_instances_tobii_compatible(window_size, buffer_size, type):
    """Convert Instances into Format Compatible for PyTrack (mimic TOBII)"""
    if type == 'pos':
        txt_dir = os.path.join(output_dir, f'pos_file_buff_{buffer_size}ms_{window_size}_sec')
        df = pd.read_csv(os.path.join(output_dir, f"pos_file_buff_{buffer_size}ms_{window_size}_sec.csv"))
    else:
        txt_dir = os.path.join(output_dir, f'neg_file_{window_size}_sec_window')
        df = pd.read_csv(os.path.join(output_dir, f"neg_file_{window_size}_sec_window.csv"))


    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)

    df['timestamp'] = df['timestamp'].astype(np.int64)

    df = df.rename(columns={'timestamp': 'Recording timestamp',
                            'scene': 'Scene Name',
                            'left_eye_pupil_position_x': 'Gaze2d_Left.x',
                            'left_eye_pupil_position_y': 'Gaze2d_Left.y',
                            'left_eye_pupil_diameter': 'PupilDiam_Left',
                            'right_eye_pupil_position_x': 'Gaze2d_Right.x',
                            'right_eye_pupil_position_y': 'Gaze2d_Right.y',
                            'right_eye_pupil_diameter': 'PupilDiam_Right'})

    df['Event value'] = 0
    df['Event message'] = 0

    width = 1440
    height = 1600
    df["Gaze2d_Left.x"] = df["Gaze2d_Left.x"].apply(
        lambda row: float(row) * width if row != "0" else row)
    df["Gaze2d_Left.y"] = df["Gaze2d_Left.y"].apply(
        lambda row: float(row) * height if row != "0" else row)
    df["Gaze2d_Right.x"] = df["Gaze2d_Right.x"].apply(
        lambda row: float(row) * width if row != "0" else row)
    df["Gaze2d_Right.y"] = df["Gaze2d_Right.y"].apply(
        lambda row: float(row) * height if row != "0" else row)

    total_instances = 0
    if type == 'pos':
        unique_instances = df[['participant', 'Scene Name']].drop_duplicates()
        for _, instance in unique_instances.iterrows():
            instance_df = df[(df['participant'] == instance['participant']) & (df['Scene Name'] == instance['Scene Name'])]
            add_event_val_and_message(instance_df)
            filename = f'participant_{instance["participant"]}_scene_{instance["Scene Name"]}_data.txt'
            full_path = os.path.join(txt_dir, filename)
            instance_df.to_csv(full_path, sep='\t', index=False)
            total_instances += 1

    if type == 'neg':
        unique_instances = df['Identifier'].drop_duplicates()
        for identifier in unique_instances:
            instance_df = df[df['Identifier'] == identifier]
            add_event_val_and_message(instance_df)
            filename = f'participant_{instance_df.iloc[0]["participant"]}_Ident_{identifier}_data.txt'
            full_path = os.path.join(txt_dir, filename)
            instance_df.to_csv(full_path, sep='\t', index=False)
            total_instances += 1

    with open(DESCRIPTIVE_STAT_PATH, 'a') as file:
        file.write(f"\tTotal unique {type} instances: {total_instances}\n")

def add_event_val_and_message(instance_df):
    """Helper Method for make_instances_tobii_compatible()"""
    instance_df.loc[instance_df.index[0], 'Event value'] = 100
    instance_df.loc[instance_df.index[-1],'Event value'] = 200
    instance_df.loc[instance_df.index[0], 'Event message'] = "MYKEYWORD 100"
    instance_df.loc[instance_df.index[-1], 'Event message'] = "MYKEYWORD 200"

def feature_extraction(window_size, buffer_size, type):
    """Extracts Features from TOBII Compatible DataFrames using PyTrack"""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    sensor_dict = {
        "EyeTracker": {
            "Sampling_Freq": 120,
            "Display_width": 1440,
            "Display_height": 1600,
            "aoi": [0, 0, 1440, 1600]
        }
    }

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    if type == 'pos':
        txt_dir = os.path.join(output_dir, f'pos_file_buff_{buffer_size}ms_{window_size}_sec')
    else:
        txt_dir = os.path.join(output_dir, f'neg_file_{window_size}_sec_window')


    feature_dfs = []

    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            participant = filename.split("_")[1]
            full_path = os.path.join(txt_dir, filename)
            csv_filename = os.path.splitext(filename)[0] + ".csv"
            csv_full_path = os.path.join(txt_dir, csv_filename)

            og_df = pd.read_csv(full_path, sep='\t')

            if not os.path.isfile(csv_full_path):
                generateCompatibleFormat(exp_path=full_path,
                                         device="tobii",
                                         stim_list_mode="NA",
                                         start="MYKEYWORD",
                                         eye="B")

            df = pd.read_csv(csv_full_path)
            stim = Stimulus(path=txt_dir, data=df, sensor_names=sensor_dict)
            stim.findEyeMetaData()
            features = stim.sensors["EyeTracker"].metadata
            MS, ms_count, ms_duration, ms_vel, ms_amp = stim.findMicrosaccades(plot_ms=True)
            fix_cnt, max_fix_cnt, avg_fix_cnt = stim.findFixationParams()
            blink_cnt, peak_blink_duration, avg_blink_duration = stim.findBlinkParams()
            pupil_size, peak_pupil, time_to_peak, pupil_AUC, pupil_slope, pupil_mean, pupil_size_downsample = stim.findPupilParams()
            saccade_count, saccade_duration, saccade_peak_vel, saccade_amplitude = stim.findSaccadeParams()

            features = {
                'participant': [participant],
                'fix_cnt': [fix_cnt],
                'max_fix_cnt': [max_fix_cnt],
                'avg_fix_cnt': [avg_fix_cnt],
                'blink_cnt': [blink_cnt],
                'peak_blink_duration': [peak_blink_duration],
                'avg_blink_duration': [avg_blink_duration],
                'pupil_size': [pupil_size],
                'peak_pupil': [peak_pupil],
                'time_to_peak': [time_to_peak],
                'pupil_AUC': [pupil_AUC],
                'pupil_slope': [pupil_slope],
                'pupil_mean': [pupil_mean],
                'pupil_size_downsample': [pupil_size_downsample],
                'microsaccade_count': [ms_count],
                'microsaccade_duration': [ms_duration],
                'microsaccade_vel': [ms_vel],
                'microsaccade_amp': [ms_amp],
                'saccade_count': [saccade_count],
                'saccade_duration': [saccade_duration],
                'saccade_peak_vel': [saccade_peak_vel],
                'saccade_amplitude': [saccade_amplitude]
            }

            new_row = pd.DataFrame(features)
            new_row.insert(1, 'study_status', og_df['study_status'][0])
            new_row.insert(2, 'scene_familiarity', og_df['scene_familiarity'][0])
            new_row.insert(3, 'recall_status', og_df['recall_status'][0])
            feature_dfs.append(new_row)
            os.remove(csv_full_path)

    warnings.resetwarnings()

    final_features_df = pd.concat(feature_dfs, ignore_index=True)

    columns_to_avg = ['pupil_size', 'pupil_size_downsample', 'saccade_duration', 'saccade_peak_vel',
                      'saccade_amplitude', 'microsaccade_duration', 'microsaccade_vel', 'microsaccade_amp']
    for column in columns_to_avg:
        avg_col_name = 'avg_' + column
        final_features_df[avg_col_name] = final_features_df[column].apply(np.mean)
    final_features_df = final_features_df.drop(columns=columns_to_avg)

    # --- Specific Features Gen ---
    # Pupil
    pupil_features = [
        'peak_pupil', 'pupil_AUC', 'pupil_slope', 'pupil_mean',
        'avg_pupil_size', 'avg_pupil_size_downsample', 'time_to_peak'
    ]

    # Fixation Features
    fixation_features = [
        'fix_cnt', 'max_fix_cnt', 'avg_fix_cnt'
    ]

    # Blink Features
    blink_features = [
        'blink_cnt', 'peak_blink_duration', 'avg_blink_duration'
    ]

    # Saccade Features
    saccade_features = [
        'saccade_count', 'avg_saccade_duration', 'avg_saccade_peak_vel', 'avg_saccade_amplitude'
    ]

    # Microsaccade Features
    microsaccade_features = [
        'microsaccade_count', 'avg_microsaccade_duration', 'avg_microsaccade_vel', 'avg_microsaccade_amp'
    ]

    #------------ 2 Way Combinations ---------------------

    # Pupil + Fixation
    pupil_fixation_features = pupil_features + fixation_features

    # Pupil + Blink
    pupil_blink_features = pupil_features + blink_features

    # Pupil + Saccade
    pupil_saccade_features = pupil_features + saccade_features

    # Pupil + Microsaccade
    pupil_microsaccade_features = pupil_features + microsaccade_features

    # Fixation + Blink
    fixation_blink_features = fixation_features + blink_features

    # Fixation + Saccade
    fixation_saccade_features = fixation_features + saccade_features

    # Fixation + Microsaccade
    fixation_microsaccade_features = fixation_features + microsaccade_features

    # Blink + Saccade
    blink_saccade_features = blink_features + saccade_features

    # Blink + Microsaccade
    blink_microsaccade_features = blink_features + microsaccade_features

    # Saccade + Microsaccade
    saccade_microsaccade_features = saccade_features + microsaccade_features

    #------------ 3 Way Combinations ---------------------

    # Pupil + Fixation + Blink
    pupil_fixation_blink_features = pupil_features + fixation_features + blink_features

    # Pupil + Fixation + Saccade
    pupil_fixation_saccade_features = pupil_features + fixation_features + saccade_features

    # Pupil + Fixation + Microsaccade
    pupil_fixation_microsaccade_features = pupil_features + fixation_features + microsaccade_features

    # Pupil + Blink + Saccade
    pupil_blink_saccade_features = pupil_features + blink_features + saccade_features

    # Pupil + Blink + Microsaccade
    pupil_blink_microsaccade_features = pupil_features + blink_features + microsaccade_features

    # Pupil + Saccade + Microsaccade
    pupil_saccade_microsaccade_features = pupil_features + saccade_features + microsaccade_features

    # Fixation + Blink + Saccade
    fixation_blink_saccade_features = fixation_features + blink_features + saccade_features

    # Fixation + Blink + Microsaccade
    fixation_blink_microsaccade_features = fixation_features + blink_features + microsaccade_features

    # Fixation + Saccade + Microsaccade
    fixation_saccade_microsaccade_features = fixation_features + saccade_features + microsaccade_features

    # Blink + Saccade + Microsaccade
    blink_saccade_microsaccade_features = blink_features + saccade_features + microsaccade_features

    #------------ 4 Way Combinations ---------------------

    # Pupil + Fixation + Blink + Saccade
    pupil_fixation_blink_saccade_features = pupil_features + fixation_features + blink_features + saccade_features

    # Pupil + Fixation + Blink + Microsaccade
    pupil_fixation_blink_microsaccade_features = pupil_features + fixation_features + blink_features + microsaccade_features

    # Pupil + Fixation + Saccade + Microsaccade
    pupil_fixation_saccade_microsaccade_features = pupil_features + fixation_features + saccade_features + microsaccade_features

    # Pupil + Blink + Saccade + Microsaccade
    pupil_blink_saccade_microsaccade_features = pupil_features + blink_features + saccade_features + microsaccade_features

    # Fixation + Blink + Saccade + Microsaccade
    fixation_blink_saccade_microsaccade_features = fixation_features + blink_features + saccade_features + microsaccade_features

    #------------ 5 Way Combination ---------------------

    pupil_fixation_blink_saccade_microsaccade_features = pupil_features + fixation_features + blink_features + saccade_features + microsaccade_features

    meta_cols = ['participant', 'study_status', 'scene_familiarity', 'recall_status']
    cols_to_keep = meta_cols + fixation_blink_saccade_microsaccade_features
    final_features_df = final_features_df[cols_to_keep]

    if type == 'pos':
        final_features_df.to_csv(
            os.path.join(output_dir, f"pos_pytrack_buff_{buffer_size}ms_{window_size}_sec.csv"), index=False
        )
    if type == 'neg':
        final_features_df.to_csv(
            os.path.join(output_dir, f"neg_pytrack_{window_size}_sec_window.csv"), index=False
        )

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    buffer_sizes = ["0", "250", "500"]
    window_sizes = ["1", "2", "3"]

    for window in window_sizes:
        for buffer in buffer_sizes:
            extract_instances(window, buffer)
            make_instances_tobii_compatible(window, buffer, 'pos')
            feature_extraction(window, buffer, 'pos')
        make_instances_tobii_compatible(window, buffer, 'neg')
        feature_extraction(window, buffer, 'neg')
