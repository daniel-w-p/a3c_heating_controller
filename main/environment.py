import time

from main import TemperatureModel


class Environment:
    def __init__(self, room_number=4, heating_source_temp=40., sunrise_time=460):
        self.temp_model = [TemperatureModel(heating_source_temp, sunrise_time, True) for _ in range(room_number)]

    def get_state(self, room_id):
        pass

    def reset(self):
        pass

    def step(self, _actions, _time):
        for i, action in enumerate(_actions):
            self.temp_model[i].step(action, _time)

    def get_values_as_strings(self):
        pass

    def get_outdoor_as_strings(self):
        pass
