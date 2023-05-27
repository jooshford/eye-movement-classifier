import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import wavfile

sample_rate, data = wavfile.read('data/greg/markers/CRCR_2FFT.wav')
markers = pd.read_csv('data/greg/markers/CRCR_2FFT_markers.csv')

directions = list()
direction_map = {
    ('left', 'L'): 'left',
    ('left', 'R'): 'centre',
    ('centre', 'L'): 'left',
    ('centre', 'R'): 'right',
    ('right', 'L'): 'centre',
    ('right', 'R'): 'right'
}
previous = 'centre'
was_previous_event = False
times = [x[0] for x in data if x[0] < 4.2]
values = [x[1] for x in data if x[0] < 4.2]
event_counter = 0
for i in range(len(times)):
    in_event = False
    for _, event in markers.iterrows():
        if times[i] >= event['start'] and times[i] <= event['end']:
            in_event = True
            if not was_previous_event:
                was_previous_event = True
                previous = direction_map[(previous, event['Action'])]
                plt.plot(times[i:i+sample_rate],
                         values[i:i+sample_rate], color='#124F7B')
                plt.xlabel('Time (s)')
                plt.ylabel('Voltage (µV)')
                plt.show()
                event_counter += 1

    if not in_event:
        was_previous_event = False

    directions.append(previous)

data = pd.DataFrame({
    'times': times,
    'values': values,
    'directions': directions
})

color_map = {
    'left': '#008001',
    'right': '#D94552',
    'centre': '#124F7B'
}

previous = 'centre'
time_segment = list()
values_segment = list()
for i in range(len(data['directions'])):
    direction = data['directions'][i]
    if direction != previous:
        plt.plot(time_segment, values_segment, color=color_map[previous])
        time_segment = list()
        values_segment = list()

    time_segment.append(data['times'][i])
    values_segment.append(data['values'][i])
    previous = direction

plt.plot(time_segment, values_segment, color=color_map[previous])

plt.xlabel('Time (s)')
plt.ylabel('Voltage (µV)')
plt.legend(['Centre', 'Right'])
plt.show()

plt.savefig('overall.png')
