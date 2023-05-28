import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_features(df):
    '''
    Plots event intervals for signals given a dataframe
    '''
    for i in df[df['event'] == 1]['T']:
        plt.axvline(i, color='g',label='Left')

    for i in df[df['event'] == 2]['T']:
        plt.axvline(i, color='g', linestyle='--')

    for i in df[df['event'] == 3]['T']:
        plt.axvline(i, color='r', label='Right')
    for i in df[df['event'] == 4]['T']:
        plt.axvline(i, color='r', linestyle='--')

    for i in df[df['event'] == 13]['T']:
        plt.axvline(i, color='orange', label='Blink')
    for i in df[df['event'] == 14]['T']:
        plt.axvline(i, color='orange', linestyle='--')
    return None


#raw data path
raw_data = f'{sys.argv[1]}/{sys.argv[2]}/RAW/{sys.argv[3]}'
#read in dataframe from csv
df = pd.read_csv(f'{raw_data}.csv')

#plot signal and event intervals
plt.plot(df['T'], df['V'])
plot_features(df)
plt.xlabel('time (s)')
plt.title(raw_data)
plt.ylabel('Voltage (a.u)')
plt.legend()
plt.show()

#FFT data path
FFT_data = f'{sys.argv[1]}/{sys.argv[2]}/FFT/{sys.argv[3]}FFT'
#read in dataframe from csv
df = pd.read_csv(f'{FFT_data}.csv')

#plot signals and event intervals
plt.plot(df['T'], df['V'])
plot_features(df)
plt.xlabel('time (s)')
plt.title(FFT_data)
plt.ylabel("Voltage (a.u.)")
plt.legend()
plt.show()
