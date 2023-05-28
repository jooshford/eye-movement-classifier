import pandas as pd
import os
import sys
rate = 10000

#directory with csv from data collection
move = f'{os.getcwd()}/{sys.argv[1]}/FORWARDS/'
#final directory to be moved into
final = f'{os.getcwd()}/{sys.argv[2]}'

for folder in os.listdir(move):
    for file in os.listdir(f'{move}/{folder}/FFT/'):
        name = file.split('.')[0]
        df = pd.read_csv(f'{move}/{folder}/FFT/{file}')
        events = df[df['event'] != 0].drop(columns='V').reset_index(drop=True)
        times = []
        for index, row in events.iterrows():
            if row['event'] == 1:
                if events['event'].loc[index+1] in [3, 4]:
                    times.append(("L",events['T'].loc[index], events['T'].loc[index+2]))
                else:
                    times.append(("L", events['T'].loc[index], events['T'].loc[index+1]))
            elif row['event'] == 3:
                if events['event'].loc[index+1] in [1, 2]:
                    times.append(("R",events['T'].loc[index], events['T'].loc[index+2]))
                else:
                    times.append(("R", events['T'].loc[index], events['T'].loc[index+1]))
            elif row['event'] == 13:
                times.append(("B", events['T'].loc[index], events['T'].loc[index+1]))
        df2 = pd.DataFrame(times).rename(columns={0:'Action', 1:'start', 2:'end'})
        df2.to_csv(f'{final}/{folder}/{folder}_{name}_markers.csv', index=False)