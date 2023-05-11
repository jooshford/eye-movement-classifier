import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from data_cleaning import process_gaussian_fft
from constants import *


def label_windows(wave, times, marker_data):
    windows = list()

    current_start = 0
    current_end = round(WINDOW_TIME * PROCESSED_SAMPLE_RATE)
    while current_end < len(wave):
        window_values_raw = wave[current_start:current_end]
        window_times = times[current_start:current_end]
        window_label = 'N'
        for _, row in marker_data.iterrows():
            window_start_time = current_start / PROCESSED_SAMPLE_RATE
            window_end_time = current_end / PROCESSED_SAMPLE_RATE
            if (row['start'] >= window_start_time and
                    row['end']-MARKER_SHORTENING_VALUE <= window_end_time):

                window_label = row['Action']

        window_values = process_gaussian_fft(window_times.tolist(),
                                             window_values_raw.tolist(),
                                             SIGMA_GAUSS)

        current_start += TRAINING_INCREMENT
        current_end += TRAINING_INCREMENT

        windows.append((window_values, window_label))

    return windows


def read_directory(wave_path, marker_path) -> list:
    wave_files = listdir(wave_path)
    windows = list()

    for wave_file in wave_files:
        if wave_file[0] == '.':
            continue

        marker_file = f'{wave_file[:-4]}FFT_markers.csv'

        wave_data = pd.read_csv(f'{wave_path}/{wave_file}')
        marker_data = pd.read_csv(f'{marker_path}/{marker_file}')

        windows.append(label_windows(wave_data['V'],
                                     wave_data['T'],
                                     marker_data))

    return windows


def filter_training_windows():
    window_sets = list()
    for _, directory in DATA_DIRECTORIES.items():
        window_sets.extend(read_directory(
            directory['values'], directory['markers']))

    label_counter = {
        'N': 0,
        'L': 0,
        'R': 0
    }

    previous = 'N'
    for i in range(len(window_sets)):
        for j in range(len(window_sets[i])):
            window = window_sets[i][j]
            values = window[0]
            label = window[1]
            window_dataframe = pd.DataFrame({'V': values})
            window_dataframe.to_csv(
                f'{WINDOWS_DIRECTORY}/{i}_{j}_{label}_{previous}.csv')

        label_counter[label] += 1
        previous = label


if __name__ == '__main__':
    filter_training_windows()
