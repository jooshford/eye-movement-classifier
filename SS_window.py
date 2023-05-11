from serial import Serial
import numpy as np
import pandas as pd
import keyboard as kb
import sys
import time

import data_cleaning

import streaming_classifier
import random_forest
import svm
import knn
from constants import *


def exit_plot():
    print('Terminated')
    sys.exit()


if __name__ == '__main__':
    kb.add_hotkey('esc', exit_plot)

    # classifiers
    print('Training classifiers...')
    training_data = pd.read_csv(TEST_TRAINING_DATA)
    models = {
        'random forest': streaming_classifier.train(random_forest.model_1(),
                                                    training_data),
        'svm': streaming_classifier.train(svm.model_3(),
                                          training_data),
        'knn': streaming_classifier.train(knn.model_10(),
                                          training_data)
    }
    print('Trained classifiers.\n')

    # bluetooth
    print('Connecting...')
    # Start communications with the bluetooth unit
    bluetooth = Serial(B_PORT, 9600)
    print('Connected.\n')
    bluetooth.flushInput()  # This gives the bluetooth a little kick

    # SpikerBox
    input_buffer_size = INCREMENT_TIME * \
        INPUT_SAMPLE_RATE + 1  # keep betweein 2000-20000

    serial = Serial(port=C_PORT, baudrate=BAUD_RATE)
    serial.timeout = input_buffer_size/INPUT_SAMPLE_RATE
    serial.set_buffer_size(rx_size=input_buffer_size)

    model = models['random forest']

    processed_v_cache = list()
    current_start_time = 0
    event_num = 0
    event_map = {
        0: 'stop',
        1: 'left',
        2: 'right'
    }

    movement_map = {
        ('stop', 'left'): 'left',
        ('stop', 'right'): 'right',
        ('left', 'left'): 'left',
        ('left', 'right'): 'stop',
        ('right', 'left'): 'stop',
        ('right', 'right'): 'right'
    }

    current_direction = 'stop'
    previous_event = 'stop'
    # this loop runs truly in parallel with the print loop, constantly checking
    while True:
        # Read data from SpikerBox into a buffer of size input_buffer_size.
        new_increment = [int(x) for x in serial.read(input_buffer_size)]
        start_time = time.time()
        # Process with above function.
        new_increment_processed = data_cleaning.process_data(new_increment)
        processed_v_cache.extend(new_increment_processed)

        if len(processed_v_cache) > WINDOW_TIME * PROCESSED_SAMPLE_RATE:
            processed_v_cache = processed_v_cache[-WINDOW_TIME *
                                                  PROCESSED_SAMPLE_RATE:]

        if len(processed_v_cache) < WINDOW_TIME * PROCESSED_SAMPLE_RATE:
            continue

        window_times = np.linspace(current_start_time,
                                   current_start_time + WINDOW_TIME,
                                   WINDOW_TIME)

        filtered_window = data_cleaning.process_gaussian_fft(
            window_times, processed_v_cache, SIGMA_GAUSS)

        down_sampled_window = [filtered_window[i] for i in range(
            len(filtered_window)) if i % TEST_DOWN_SAMPLE_RATE == 0]
        event_num = streaming_classifier.classify(
            model,
            filtered_window,
            int(event_num == 1),
            int(event_num == 2))

        event = event_map[event_num[0]]
        if event != previous_event and event != 'stop':
            current_direction = movement_map[(current_direction, event)]

        previous_event = event
        # encode to utf-8
        direction_bytes = current_direction.encode('utf_8')
        # communicate to bluetooth
        bluetooth.write(direction_bytes)
        end_time = time.time()

        print(f'{current_direction} (execution time: {(end_time - start_time):.4f})')
