import math
from .data_dict import DataDict
# TODO : remember to penalizing to often heating switching and floor heating over desired max temp


class TemperatureModel:
    """
    This class is used to model temperatures using numerical methods.

    Attributes:
        min_switch_time (int): constant.
        max_floor_temperature (float): constant.
        min_out_temperature (float): constant.
        max_out_temperature (float): constant.
        thermal_coef (float): constant coefficient of thermal transmittance.
        alpha (float): constant coefficient - floor heating rate factor.
        beta (float): constant coefficient - floor cooling rate factor.
        self.outdoor_temperature (float): approximated from sinus.
        self.indoor_temperature (float): numerically approximated.
        self.heating_temperature (float): numerically approximated.
        self.heating_source_temp (float): some constant temperature.

    Methods:
        calculate_outdoor_temperature: returns the outdoor temperature (float).
        calculate_indoor_temperature: returns the indoor temperature (float).
        calculate_heating_temperature: returns the floor temperature (float).
    """
    min_switch_time = 10
    max_floor_temperature = 27.
    min_out_temperature = -10.
    max_out_temperature = 5.
    thermal_coef = 0.5
    alpha = 0.0008 * 60  # because this for minute
    beta = 0.0016 * 60  # because this for minute

    def __init__(self, heating_source_temp=40., sunrise_time=460, sub_minute_for_day=True):
        """
        Constructor.

        Args:
            heating_source_temp (float): temperature reached by the installation.
            sunrise_time (int): sunrise time in minutes.
            sub_minute_for_day (bool): if True the sunrise time will be moved.
        """
        self.outdoor_temperature = 0.
        self.indoor_temperature = 20.
        self.heating_temperature = 27.
        self.heating_source_temp = heating_source_temp
        self.sunrise_time = sunrise_time
        self.sub_minute_for_day = sub_minute_for_day
        self.heating_source_on = False
        self.last_switch_time = 0
        self.data_dictionary = DataDict()

    def calculate_outdoor_temperature(self, time: int):
        """
        Simple sinus based simulation. Not really accurate.

        Args:
            time (int): time in minutes.
        """
        sunrise_time = self.sunrise_time
        if self.sub_minute_for_day:  # TODO this could be done better (because in real it is not a minute per day)
            sunrise_time -= time % 1440  # minutes per day
        self.outdoor_temperature = (self.min_out_temperature +
                                    (self.max_out_temperature - math.fabs(self.min_out_temperature)) * math.sin(
                                        (2 * math.pi / 1440) * (time - sunrise_time)))

    def calculate_indoor_temperature(self, time: int):
        """
        Numerical approximation.
        From equation: \n
        dT/dx = H(t) - k*(T(t) - To(t)) \n
        where:
        - H(t) is the heating temperature.
        - k is constant coefficient of thermal transmittance (building <-> outdoor).
        - To(t) is outdoor temperature.

        Args:
            time (int): time in minutes.
        """
        self.calculate_outdoor_temperature(time)

        self.indoor_temperature += self.heating_temperature - self.thermal_coef * (
                self.indoor_temperature - self.outdoor_temperature)

    def calculate_heating_temperature(self):
        """
        Numerical approximation.
        From equation: \n
        dH/dx = on*a*(Hmax - H(t)) - B*(H(t) - T(t)) \n
        where:
        - on is casted to int flag of heating on / off (1 / 0)
        - Hsrc is the source heating temperature.
        - H(t) is the heating temperature.
        - a (alpha) is the coefficient of the speed of floor heating,
        - B (beta) is the coefficient of the speed of floor cooling,
        - T(t) is the ambient temperature inside the building.
        """
        self.heating_temperature += (int(self.heating_source_on) *
                self.alpha * (self.heating_source_temp - self.heating_temperature) -
                self.beta * (self.heating_temperature - self.indoor_temperature)
        )

    def calculate_temperatures(self, time: int):
        self.calculate_indoor_temperature(time)
        self.calculate_heating_temperature()

        self.save_values_to_dict(time)

    def step(self, action: bool, time: int):
        if self.heating_source_on is not action:
            self.switch_heating_source(time)
        self.calculate_temperatures(time)
        return self.get_values()

    def save_values_to_dict(self, time: int):
        self.data_dictionary.add_data(time, self.outdoor_temperature, self.indoor_temperature, self.heating_temperature, self.heating_source_on)

    def switch_heating_source(self, time: int):
        self.last_switch_time = time
        self.heating_source_on = not self.heating_source_on

    def get_values(self):
        pass

    # TODO  - for now I use simple Euler method
    def runge_kutta_step(self, Yn, k1):
        """
        Runge-Kutta method for one step of temperature update.\n
        'h' - time step is one (update every minute)\n
        k2 = f(t+h, Yn + h*k1)\n
        Yn+1 = Yn + 1/2 * (k1 + k2)  (h/2 = 1/2)

        Args:
            Yn (float): last (temperature) value.
            k1 (float): first step value.

        Returns:
            float: next step value.
        """
        pass
