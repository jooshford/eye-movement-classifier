import pandas as pd
from filter_training_windows import filter_training_windows
from write_training_data import write_training_data
from write_accuracy_results import write_accuracy_results
from write_time_results import write_time_results
from models import models, top_models
from constants import *


def setup_from_windows():
    write_training_data()


def setup_from_start():
    filter_training_windows()
    write_training_data()


if __name__ == "__main__":
    training_data = pd.read_csv(f'{TRAINING_DIRECTORY}/1.csv')
    for model_name, model_function in top_models.items():
        write_accuracy_results(model_function,
                               model_name,
                               1,
                               training_data,
                               num_repeats=10)
