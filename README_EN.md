# Smart Temperature Management System

## Languages

- [English](README_EN.md)
- [Polski](README.md)

## Description
The project aims to create a smart temperature management system in a building using machine learning techniques, specifically the A3C (Asynchronous Advantage Actor-Critic) model. The system's goal is to automatically regulate heating to maintain an optimal temperature set by the user, while minimizing the frequency of heating switches.

## Components
The project consists of several key components:

### Environmental Simulation Module
This module, based on differential equations, simulates the dynamics of temperature in the building. Data on temperatures and heating status are recorded in a table using Pandas, which allows for further analysis.

### Differential Equations of the Model

The equation describing the change in indoor temperature over time is defined as:

\[ \frac{dT}{dt} = \frac{1}{C} \left(\eta \cdot (H(t) - T(t)) - k \cdot A \cdot (T(t) - T_{\text{ext}}(t))\right) \]

### Legend:

- \( T(t) \): Indoor temperature of the room at time \( t \) [°C].
- \( H(t) \): Temperature of the floor (i.e., the heat source) at time \( t \) [°C].
- \( T_{\text{ext}}(t) \): Outdoor temperature at time \( t \) [°C].
- \( \eta \): Efficiency coefficient of heat transfer from the floor to the room air [W/m²K].
- \( k \): Heat transfer coefficient through the building's walls [W/m²K].
- \( A \): Total surface area of the room's exterior walls [m²].
- \( C \): Thermal capacity of the room, indicating the amount of energy needed to warm up the entire room air by one degree Celsius [J/K].

### Explanation of the Equation:

This equation describes how quickly the temperature in the room changes in response to the operation of the floor heating system and the heat exchange with the external environment. The coefficient \( \eta \) measures how effectively heat is transferred from the floor to the room air, and the coefficient \( k \) reflects how quickly heat escapes from the room through the external walls. The thermal capacity \( C \) describing how much heat the room can store.

Equation describing the change in floor temperature H(t) over time:

```math
\[ \frac{dH}{dt} = \alpha (H_{\text{max}} - H(t)) - \beta (H(t) - T(t)) \]
```

where:

- \(H_{\text{max}}\) is a constant representing the maximum temperature the floor can reach,
- \(\alpha\) is the coefficient of the speed of floor heating,
- \(\beta\) is the coefficient of the speed of floor cooling,
- \(T(t)\) is the ambient temperature inside the building.

### Data Analysis and Visualization
Using the Matplotlib library, data are analyzed and visualized in the form of charts, allowing for the assessment of the heating system's efficiency and temperature management strategies.

### Interactive Visualization in Pygame
An additional module written in Pygame provides interactive visualization of the model's operation in real-time, with the possibility of accelerating the passage of time. This allows for observing the effects of the system's operation in an accessible visual form.

## Technologies
- Python
- TensorFlow
- Pandas
- Matplotlib
- Pygame 

## Project Launch

### Run simulation
```python run_simulator.py```

### Run training
```python run_training.py```

## License
[LICENSE](LICENSE)

---

The project is a combination of machine learning, physics, and data visualization, aimed at improving the energy efficiency of buildings through smart temperature management.