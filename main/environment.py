import time

from main import TemperatureModel


class Environment:
    """
    This class is used to create the environment based on the TemperatureModel objects.
    """
    def __init__(self, rooms_desired_temp: list, heating_source_temp=40., sunrise_time=460):
        rooms_num = len(rooms_desired_temp)
        self.rooms_desired_temp = rooms_desired_temp
        self.temp_model = [TemperatureModel(heating_source_temp, sunrise_time, True) for _ in range(rooms_num)]
        self.time = 0

    def get_state(self, room_id):
        """
        This method is used to get the state of one room.
        Args:
            room_id: (int) 0 to len()-1
        Returns:
            tuple of (temperatures, heating_source, desired_temp and time)
        """
        in_values = self.temp_model[room_id].get_in_values()
        out_values = self.temp_model[room_id].get_out_values()
        state = *in_values, *out_values, self.rooms_desired_temp[room_id], self.time
        return state

    def reset(self):
        """
        This method is used to reset the time and return states of the environment.
        Returns:
            list of tuples of (temperatures, heating_source, desired_temp and time)
        """
        self.time = 0
        states = []
        for tm, rdt in zip(self.temp_model, self.rooms_desired_temp):
            states.append((*tm.get_in_values(), *tm.get_out_values(), rdt, self.time))
        return states

    def step(self, actions, time_step):
        """
        This method is used to perform one step of the environment based on the action taken.

        Args:
            actions:     list of actions (one for each room)
            time_step:   int (adding minutes)

        Returns:
            list of tuples of (temperatures, heating_source, desired_temp and time)
        """
        states = []
        self.time += time_step
        for i, action in enumerate(actions):
            self.temp_model[i].step(action, self.time)
            states.append(self.get_state(i))
        return states, self.get_penalty()

    def get_values(self):
        values = []
        for tm, rdt in zip(self.temp_model, self.rooms_desired_temp):
            values.append((rdt, *tm.get_in_values()))

        values.append((*self.temp_model[0].get_out_values(), self.time))
        return values

    def get_penalty(self):
        penalties = []
        for tm, rdt in zip(self.temp_model, self.rooms_desired_temp):
            penalties.append((tm.indoor_temperature - rdt)**2 + (max(0, tm.heating_temperature - tm.max_floor_temperature))**2)  # max - penalize only when the floor temperature exceeds the max
        return penalties


