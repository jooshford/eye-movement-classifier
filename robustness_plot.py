import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

values = np.repeat([1, 0, 1, 0, 1, 0, 1, 0], [78, 22, 75, 25, 96, 4, 91, 9])
event_type = np.repeat(['Blink', 'Directional'], [200, 200])
tester_type = np.repeat(
    ['Trained', 'Untrained', 'Trained', 'Untrained'], [100, 100, 100, 100])

data = pd.DataFrame({
    'Accuracy': values,
    'Event Type': event_type,
    'User Type': tester_type
})

sns.catplot(data,
            kind='bar',
            x='Event Type',
            y='Accuracy',
            hue='User Type',
            palette=sns.color_palette(['#D94552', '#124F7B']),
            errorbar=None)
plt.title('Comparing Metrics Between Trained and Untrained Users')
plt.show()
