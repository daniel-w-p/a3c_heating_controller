
class TwoStateSwitch:
    """
    This class represents a two-state switch with simple algorithm based on hysteresis.
    It can be helpful to compare with quality of model performance.
    """
    min_switch_time = 10

    def __init__(self, desired_temp=22.):
        self.state = False
        self.hysteresis = 0.5
        self.desired_temp = desired_temp

    def switch(self, state):
        self.state = state

    def choose_simulation_action(self, temperature):
        if temperature > self.desired_temp + self.hysteresis:
            self.switch(False)
        elif temperature < self.desired_temp - self.hysteresis:
            self.switch(True)
        return self.state

    def get_state(self):
        return self.state

