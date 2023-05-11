from constants import *
from feature_analysis import calculate_features, feature_list
import pandas as pd
from read_directory import read_directory


def generate_training_data(windows, labels, previous, down_sample_rate):
    labels_factor = [['N', 'L', 'R'].index(char) for char in labels]

    previous_L = list()
    previous_R = list()
    for x in previous:
        previous_L.append(int(x == 'L'))
        previous_R.append(int(x == 'R'))

    features = dict()
    for feature in feature_list:
        features[feature] = list()

    features['previous_L'] = previous_L
    features['previous_R'] = previous_R
    features['label'] = labels_factor

    for window in windows:
        down_sampled_window = [window[i] for i in range(
            len(window)) if i % down_sample_rate == 0]
        window_features = calculate_features(down_sampled_window)
        for name, value in window_features.items():
            features[name].append(value)

    return pd.DataFrame(features)


def write_training_data(down_sample_rate):
    print(f'Writing data for down sample rate {down_sample_rate}...')
    windows, labels, previous = read_directory(WINDOWS_DIRECTORY)
    training_data = generate_training_data(
        windows, labels, previous, down_sample_rate)

    destination = f'{TRAINING_DIRECTORY}/{down_sample_rate}.csv'
    training_data.to_csv(destination)

    return training_data
