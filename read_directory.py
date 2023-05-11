import pandas as pd
from os import listdir


def read_file(directory: str, file_name: str) -> dict:
    wave_data = pd.read_csv(f'{directory}/{file_name}')
    file_info = file_name.split('.')[0].split('_')
    values = wave_data['V'].values
    file = int(file_info[0])
    index = int(file_info[1])
    label = file_info[2]
    previous = file_info[3]

    return (file, index, values, label, previous)


def read_directory(path: str) -> list:
    files = listdir(path)
    file_nums = list()
    indexes = list()
    values = list()
    labels = list()
    previous = list()
    for file in files:
        if file[0] == '.':
            continue

        file_num, index, file_values, file_label, file_previous = read_file(
            path, file)

        file_nums.append(file_num)
        indexes.append(index)
        values.append(file_values)
        labels.append(file_label)
        previous.append(file_previous)

    return (file_nums, indexes, values, labels, previous)
