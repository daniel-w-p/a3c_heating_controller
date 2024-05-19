# Smart Temperature Management System

## Languages

- [English](README_EN.md)
- [Polski](README.md)

## Description
The project aims to create an AI model for a building temperature management system using machine learning techniques, specifically the A3C (Asynchronous Advantage Actor-Critic) model. The system is designed to automatically regulate heating to maintain the optimal temperature set by the user, while minimizing the frequency of heating switching and deviation from the target temperature. The project allows for simulating the exchange of heat between the building and its surroundings, but an accurate representation of the physics of this process is NOT (for now) the subject of this project.

For comparative purposes, a simple two-state model has been created, which activates heating when the temperature falls below a set level (1°C hysteresis). Comparing these models in the prepared environment is possible thanks to the generated data available at: ```data/plots/``` and ```data/table/```

### What has been achieved:

The trained model switches heating between the on/off states (e.g., a two-state valve). The model allows for analyzing the situation (state) for any number of rooms with similar characteristics and determining the best action (turn on / turn off). The model operates on data collected over 7 hours at 10-minute intervals (42 vectors). These data are treated as the current state in accordance with the Markov decision process.

### What can be done:

- Refine the simulator to better reflect reality (heat exchange, open windows, sudden weather changes).
- Try to reduce the model size, or increase its accuracy.
- Retrain the model to regulate heating in a smooth manner (percentage of opening: 0-100%).

## Components
The project consists of several key components:

### Environmental Simulation Module
This module, based on differential equations, simulates the temperature dynamics in a building. Temperature data and heating status are transmitted to the model's working environment, either a simulator or a silent mode. The latter saves the data to a file and generates a graph.

### Differential Equations of the Model

The equation describing the change in indoor temperature over time is defined as:

$$
\frac{dT}{dt} = \frac{1}{C} \left(\eta \cdot A_{\text{f}} \cdot (H(t) - T(t)) - k \cdot A_{\text{w}} \cdot (T(t) - T_{\text{zew}}(t))\right) 
$$

### Legend:

- \( T(t) \): Indoor temperature of the room at time \( t \) [°C].
- \( H(t) \): Temperature of the floor (i.e., the heat source) at time \( t \) [°C].
- \( T_ext(t) \): Outdoor temperature at time \( t \) [°C].
- \( \eta \): Efficiency coefficient of heat transfer from the floor to the room air [W/m²K].
- \( k \): Heat transfer coefficient through the building's walls [W/m²K].
- \( A_w \): Total surface area of the room's exterior walls [m²].
- \( A_f \): Surface area of the room's floor [m²].
- \( C \): Thermal capacity of the room, indicating the amount of energy needed to warm up the entire room air by one degree Celsius [J/K].

### Explanation of the Equation:

This equation describes how quickly the temperature in the room changes in response to the operation of the floor heating system and the heat exchange with the external environment. The coefficient \( \eta \) measures how effectively heat is transferred from the floor to the room air, and the coefficient \( k \) reflects how quickly heat escapes from the room through the external walls. The thermal capacity \( C \) describing how much heat the room can store.

Equation describing the change in floor temperature H(t) over time:

$$
\frac{dH}{dt} = \alpha (H_{\text{max}} - H(t)) - \beta (H(t) - T(t))
$$

where:

- \(H_{\text{max}}\) is a constant representing the maximum temperature the floor can reach,
- \(\alpha\) is the coefficient of the speed of floor heating,
- \(\beta\) is the coefficient of the speed of floor cooling,
- \(T(t)\) is the ambient temperature inside the building.

### Data Analysis and Visualization

For the silent mode, the Pandas and Matplotlib libraries were used. The data are analyzed and visualized in the form of charts (one room), which allows for evaluating the effectiveness of the heating system and temperature management strategies.

### Interactive Visualization in Pygame
An additional module written in Pygame provides interactive visualization of the model's operation in real-time, with the possibility of accelerating the passage of time. This allows for observing the effects of the system's operation in an accessible visual form.

## Technologies
- Python
- TensorFlow
- Numpy
- Pandas
- Matplotlib
- Pygame 

## Project Launch

The environment must be prepared, containing libraries for the technologies mentioned above, as well as a copy of the project.

One of the ways:

```bash
# Get the repo
git clone https://github.com/daniel-w-p/a3c_heating_controller.git

cd a3c_heating_controller

# Install requirements by pip
pip install -r requirements.txt
```

### Run simulation
```python run_simulator.py```

### Run silent mode
```python run_silent_mode.py```

### Run training
```python run_training.py```

## License
[LICENSE](LICENSE)

---

The project is a combination of machine learning, physics, and data visualization, aimed at preparing a model for effective temperature management.