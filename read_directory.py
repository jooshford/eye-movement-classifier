import pandas as pd
from os import listdir


def read_file(directory: str, file_name: str) -> dict:
    wave_data = pd.read_csv(f'{directory}/{file_name}')
    values = wave_data['V'].values
    label = file_name.split('_')[0]
    previous = file_name.split('_')[2].split('.')[0]

    return (values, label, previous)


def read_directory(path: str) -> list:
    files = listdir(path)
    values = list()
    labels = list()
    previous = list()
    for file in files:
        if file[0] == '.':
            continue

        file_values, file_label, file_previous = read_file(path, file)

        values.append(file_values)
        labels.append(file_label)
        previous.append(file_previous)

    return (values, labels, previous)
