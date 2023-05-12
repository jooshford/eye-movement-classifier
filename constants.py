DATA_DIRECTORIES = {
    'garry': {
        'values': 'data/garry/values',
        'markers': 'data/garry/markers'
    },
    'greg': {
        'values': 'data/greg/values',
        'markers': 'data/greg/markers'
    },
    'yifei': {
        'values': 'data/yifei/values',
        'markers': 'data/yifei/markers'
    }
}

INPUT_SAMPLE_RATE = 20000.0
PROCESSED_SAMPLE_RATE = 10000
SIGMA_GAUSS = 25

DOWN_SAMPLE_RATES = [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
MARKER_SHORTENING_VALUE = 0.2
TRAINING_INCREMENT = 1000
TRAINING_DIRECTORY = 'data/training'
WINDOWS_DIRECTORY = 'data/windows'

INCREMENT_TIME = 0.1
WINDOW_TIME = 1

CV_NUM_FOLDS = 5
RESULTS_DIRECTORY = 'data/results'
TIMES_DIRECTORY = 'data/times'

PLOT_LINE_WIDTH = 0.5

B_PORT = 'COM9'
C_PORT = 'COM7'
BAUD_RATE = 230400

TEST_TRAINING_DATA = 'data/training/1024.csv'
TEST_DOWN_SAMPLE_RATE = 1024
