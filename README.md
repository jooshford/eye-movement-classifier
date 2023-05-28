# Eye Movement Classification using EOG Signals and Machine Learning

## Requirements

### Python

The project was run using Python version 3.10, and uses the following libraries:

- matplotlib
- seaborn
- numpy
- pandas
- time
- sklearn
- scipy
- keyboard
- serial
- os

### R

The report was compiled using Quarto Markdown, and requires the following packages:

- reticulate
- kable

## Execution Instructions

In this repository are already-collected datasets, already-calculated training datasets, and cached model evaluation metrics. However, if you wanted to do this from scratch with new data, you could delete these from your local system, and do the following:

### Generating Data

Ensure electrode positions are such that a horizontal line is formed through the electrodes and eyes. Folders must be setup in the following: User_name / Eye Action and within Eye Action there are two directories RAW and FFT. When calling the python script SS.py we need to specify command line arguments that will tell the script where to save the data. The script should be run as:

```bash
python3 SS.py [user_name] [eye_action] [trial_number]
```

For example:

```bash
python3 SS.py John_doe CL 42
```

During data collection, numbers will print down the terminal indication that the script is running. To mark a relative left look (meaning from center to left or from right to center) press 9, relative right press 0 and a blink press b. The timing of the keypresses does not need to be with the action, but they do need to retain the order. To stop data collection, press `esc`. The output of the script will be placed in the RAW and FFT directories of the corresponding Eye Action. The format of the data will be in a csv with the first column labeled ‘T’ indicating time, second column ‘V’ indicating the voltage and ‘event’ indicating the event. The events are labeled with number:

| Event type            | Label |
| --------------------- | ----- |
| Start event for left  | 1     |
| End event for left    | 2     |
| Start event for right | 3     |
| End event for right   | 4     |
| Start event for blink | 13    |
| End event for blink   | 14    |
| Otherwise             | 0     |

(put in table if you like josh) start event for left - 1, end event for left - 2, start event for right - 3, end event for right - 4, start event for blink - 13, end event for blink - 14, 0 otherwise.

### Generating Event Markers

To generate event intervals from this data, create a new directory that is in the same root as the User_name directory and the convert_csv.py script, labeled User_name_markers. Within this directory, create new directories named after the Eye Action. Run the script with the following command line arguments: python3 convert_csv.py User_name User_name_markers. The event intervals should be specified in time order and in a csv format.

### Splitting Data Into Windows

First, go to `constants.py`, and update the `DATA_DIRECTORIES`, to include the signal values and corresponding event markers. Then, create a new folder with the path `data/windows` and run the following:

```bash
python3 filter_training_windows.py
```

(NOTE: This generates a large amount of data, so be careful)

### Generating Training Datasets

In your terminal, run the following command:

```bash
python3 write_training_data.py
```

This will generate a training dataset for a variety of different downsampling rates, which will be used to train the classification algorithms on. Then, we create the training datasets with the different feature selection algorithms applied to it, using the following:

```bash
python3 feature_selection.py
```

### Evaluating Models

To evaluate the performance of the models, run the following:

```bash
python3 write_accuracy_results.py
python3 write_time_results.py
```

## Using the Classifier

To begin using the streaming classifier, run:

```bash
python3 SS_windows.py
```

Then, wait for the classifications to begin showing up in the terminal.

# Arduino Code

The code to upload to the Arduino is located in `drtive.ino`.
