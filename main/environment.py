import math

import numpy as np

from main import TemperatureModel


class Environment:
    """
    This class is used to create the environment based on the TemperatureModel objects.

    Attributes:
        rooms_num (int): the number of rooms in the environment
        rooms_desired_temp (list): one temperature per room (e.g. [21., 20.5, 19.5, 20.5])
        temp_model (TemperatureModel): the temperature model (numerical approximation of temperatures change per room)
        state_series (list): list of timeseries of states vectors
        time (int): the time of the environment running
    """
    T_DAY = 1440
    T_HALF_DAY = T_DAY // 2

    def __init__(self, rooms_desired_temp: list, with_random=True, heating_source_temp=40., sunrise_time=460):
        """
        Constructor
        Args:
            rooms_desired_temp (list): one temperature per room (e.g. [21., 20.5, 19.5, 20.5])
            heating_source_temp (float): treated as constant
            sunrise_time: (int): in minutes
        """
        random_val = np.random.uniform(-0.25, 0.25) if with_random else 0
        self.rooms_num = len(rooms_desired_temp)
        self.rooms_desired_temp = rooms_desired_temp
        self.temp_model = [TemperatureModel(rooms_desired_temp[i]+random_val, heating_source_temp, sunrise_time, True) for i in range(self.rooms_num)]
        self.state_series = []
        self.time = 0

        self.reset()

    def get_state(self, room_id):
        """
        This method is used to get the state of one room.
        Args:
            room_id: (int) 0 to len()-1
        Returns:
            tuple of (temperatures, heating_source, desired_temp and time)
        """
        theta = (2 * math.pi * (self.time % self.T_DAY)) / self.T_DAY
        in_values = self.temp_model[room_id].get_in_values(self.time)
        out_values = self.temp_model[room_id].get_out_values()
        state = *in_values, *out_values, self.rooms_desired_temp[room_id], math.sin(theta), math.cos(theta)
        return state

    def reset(self):
        """
        This method is used to reset the time and return states of the environment.
        Returns:
            list of tuples of (temperatures, heating_source, desired_temp and time)
        """
        self.time = 0
        states = []
        actual_states = []
        self.state_series = []
        theta = (2 * math.pi * (self.time % self.T_DAY)) / self.T_DAY
        for tm, rdt in zip(self.temp_model, self.rooms_desired_temp):
            tm.reset()
            states.append((*tm.get_in_values(self.time), *tm.get_out_values(), rdt, math.sin(theta), math.cos(theta)))
        states = np.array(states)
        # duplicating a single vector into a given array size
        self.state_series = np.tile(states[:, np.newaxis, np.newaxis, :], (1, 10, 42, 1))  # 10min * 42 = 7h
        for i in range(self.rooms_num):
            actual_states.append(self.state_series[i, 0, :, :])

        return np.array(actual_states)

    def step(self, actions, time_step):
        """
        This method is used to perform one step of the environment based on the action taken.

        Args:
            actions:     list of actions (one for each room)
            time_step:   int (adding minutes)

        Returns:
            list of tuples of (temperatures, heating_source, desired_temp and time)
        """
        self.time += time_step
        actual_states = []
        for i, action in enumerate(actions):
            self.temp_model[i].step(bool(action), self.time)
            # adding vector on last position in list and remove first one but doing this once per 10 min.
            # that makes range of 8 hours (10min * 42 vectors in matrix)
            new_states = np.vstack([self.state_series[i][self.time % 10][1:], self.get_state(i)])
            self.state_series[i][self.time % 10] = new_states
            actual_states.append(new_states)

        return np.array(actual_states), self.get_penalty()

    def get_values(self):
        values = []
        for tm, rdt in zip(self.temp_model, self.rooms_desired_temp):
            values.append((rdt, *tm.get_in_values(self.time)))

        values.append((*self.temp_model[0].get_out_values(), self.time))
        return values

    def get_time(self):
        return self.time

    def get_penalty(self):
        penalties = []
        for tm, rdt in zip(self.temp_model, self.rooms_desired_temp):
            switch_frequency_penalty = tm.get_switch_heating_difference(self.time)**2
            penalties.append(100 - (4 * (tm.indoor_temperature - rdt))**2 -
                             (4 * (max(0, tm.heating_temperature - tm.max_floor_temperature)))**2 -  # max() - penalize only when the floor temperature exceeds the max
                             max(0, 10 - switch_frequency_penalty))  # max() - penalize only when the switch time is less than 10 minutes
        return np.array(penalties)


