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
Equation describing the change in internal temperature T(t) over time:

```math
\[ \frac{dT}{dt} = h(t) - k \cdot (T(t) - T_{\text{out}}(t)) \]
```

where:

- \(h(t)\) represents the heating impact (can be a function dependent on the floor temperature 
- \(k\) is the coefficient of heat penetration through the building's walls,
- \(T_{\text{out}}(t)\) is the outdoor temperature.
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