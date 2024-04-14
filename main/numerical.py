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
        k_coef (float): constant coefficient of thermal transmittance.
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
        calculate_heating_temperature: numerical approximation for all temperatures based on given time.
    """
    min_switch_time = 10
    max_floor_temperature = 27.
    min_out_temperature = -10.
    min_max_temp_distance = 11.

    alpha = 0.0006 * 60  # because this is for minute
    beta = 0.0012 * 60  # because this is for minute
    # room size: 4 * 4 * 2.6 # there are always two outside walls 2 * 4 * 2.6 = 20.8
    k_coef = 20.8 * 0.5 / 50000  # TODO IMPROVE this should depend on the size of the room
    mu_coef = 15 / 50000  # TODO IMPROVE this should depend on the size of the room

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
        self.heating_temperature = 25.
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
        # TODO later possible add support for the influence of cloud cover on temperature
        sunrise_time = self.sunrise_time
        half_temp_diff = self.min_max_temp_distance / 2
        day_time = 420
        if self.sub_minute_for_day and sunrise_time - time // 1440 > 300 and day_time < 600:  # TODO this could be done better (because in real it is not 2 min. per day)
            sunrise_time -= time // 720    # 2 minutes per day
            day_time += time // 360         # if sunrise is one minute earlier and sunset is later that gives 2 min.
        self.outdoor_temperature = (self.min_out_temperature + half_temp_diff + (day_time / 420) - 1 +
                                    half_temp_diff * math.sin((2 * math.pi / 1440) * (time % 1440 - sunrise_time)))

    def calculate_heating_effect(self, time: int):
        """
        Numerical approximation.
        From equation: \n
        h(t) = mu * (H(t) - T(t)) \n
        where:
        - mu is coefficient of heating rate factor.\n
        - H(t) is the heating temperature.
        - T(t) is the ambient temperature inside the building.

        Args:
            time (int): time in minutes.

        Returns:
            float: heating effect.
        """
        return self.mu_coef * (self.heating_temperature - self.indoor_temperature)

    def calculate_indoor_temperature(self, time: int):
        """
        Numerical approximation.
        From equation: \n
        dT/dx = h(t) - k*(T(t) - To(t)) \n
        where:
        - h(t) function representing the effect of heating.
        - k is constant coefficient of thermal transmittance (building <-> outdoor).
        - To(t) is outdoor temperature.\n
        h(t) = mu * (H(t) - T(t)) \n
        where:
        - mu is coefficient of heating rate factor.\n
        - H(t) is the heating temperature.
        - T(t) is the ambient temperature inside the building.

        Args:
            time (int): time in minutes.
        """
        self.calculate_outdoor_temperature(time)
        h = self.calculate_heating_effect(time)

        self.indoor_temperature += h - self.k_coef * (
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
        """
        Numerical approximation for all temperatures based on given time.
        """
        self.calculate_indoor_temperature(time)
        self.calculate_heating_temperature()

        self.save_values_to_dict(time)

    def step(self, action: bool, time: int):
        """
        Numerical approximation for all temperatures based on given action and time.
        """
        if self.heating_source_on is not action:
            self.switch_heating_source(time)
        self.calculate_temperatures(time)

    def save_values_to_dict(self, time: int):
        """
        Save data to dictionary to further use. (tables, plots, etc).
        """
        self.data_dictionary.add_data(time, self.outdoor_temperature, self.indoor_temperature, self.heating_temperature, self.heating_source_on)

    def switch_heating_source(self, time: int):
        self.last_switch_time = time
        self.heating_source_on = not self.heating_source_on

    def get_in_values(self):
        """
        Get all indoor temperature values and heating source on/off status

        Returns:
            (tuple) of floats
        """
        return self.indoor_temperature, self.heating_temperature, self.heating_source_on

    def get_out_values(self):
        """
        Get all outdoor temperature

        Returns:
            (tuple) of floats
        """
        return (self.outdoor_temperature,)

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
