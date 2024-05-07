import os

import pandas as pd
from matplotlib import pyplot as plt


class DataDict:
    """
    This is helper class to save all data about environment
    """
    def __init__(self):
        self.data = {
            'time': [],
            'outdoor_temp': [],
            'indoor_temp': [],
            'heating_temp': [],
            'heating_on': []
        }

    def add_data(self, time: int, outdoor_temp: float, indoor_temp: float, heating_temp: float, heating_on: bool):
        self.data['time'].append(time)
        self.data['outdoor_temp'].append(outdoor_temp)
        self.data['indoor_temp'].append(indoor_temp)
        self.data['heating_temp'].append(heating_temp)
        self.data['heating_on'].append(int(heating_on))

    def save_data(self, name: str = 'temp', path: str = 'data/table'):
        df = pd.DataFrame(self.data)

        if not os.path.exists(path):
            os.makedirs(path)

        file = os.path.join(path, name + '.csv')
        df.to_csv(file, index=False)

    def plot_data(self, name: str = 'temp', path: str = 'data/plots'):
        df = pd.DataFrame(self.data)

        fig, ax1 = plt.subplots()

        # Temperatures
        ax1.plot(df['time'], df['outdoor_temp'], label='Outdoor')
        ax1.plot(df['time'], df['indoor_temp'], label='Indoor')
        ax1.plot(df['time'], df['heating_temp'], label='Heating')
        ax1.set_ylabel('Temperatures')
        ax1.set_ylim(-15, 40)  # Min - Max temperature

        # Heating On/Off
        ax2 = ax1.twinx()  # additional OY
        ax2.scatter(df['time'], df['heating_on'] * 49 - 12, color='red', label='On/Off', marker='o')
        ax2.set_ylabel('Heating on')
        ax2.set_ylim(-15, 40)  # same as Min - Max temperature

        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.title('Temperature with heating on/off')

        if not os.path.exists(path):
            os.makedirs(path)

        file = os.path.join(path, name + '.png')

        plt.savefig(file, dpi=300)

        plt.close(fig)
