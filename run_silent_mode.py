import sys

import numpy as np

from main import Environment, TwoStateSwitch, DataDict
from ai import A3CModel, Agent


COUNT_ROOMS = 1
TIME_STEPS = 2880  # one and half day


def make_step(model, env, state, ai_model=False):
    if ai_model:
        actions = Agent.choose_simulation_all_action(state, model, False)
    else:
        actions = [model.choose_simulation_action(state[0][-1][0]) for _ in range(COUNT_ROOMS)]

    state, _ = env.step(actions, 1)  # 1 - one minute
    return state


def run_silent_mode():
    data_simple = DataDict()
    data_ai = DataDict()
    # environment
    rooms_desired_temp = 21.5
    env_simple = Environment([rooms_desired_temp], False)
    env_ai = Environment([rooms_desired_temp])

    model_two_state = TwoStateSwitch(rooms_desired_temp)
    model_ai = A3CModel()
    # Get state
    states_simple = env_simple.reset()
    states_ai = env_ai.reset()
    # Lazy build A3C model
    model_ai(states_ai)

    Agent.load_model(model_ai)

    for step in range(TIME_STEPS):
        data_simple.add_data(step, states_simple[0][-1][5], states_simple[0][-1][0], states_simple[0][-1][1], states_simple[0][-1][2])
        data_ai.add_data(step, states_ai[0][-1][5], states_ai[0][-1][0], states_ai[0][-1][1], states_ai[0][-1][2])
        states_ai = make_step(model_ai, env_ai, states_ai, ai_model=True)
        states_simple = make_step(model_two_state, env_simple, states_simple, ai_model=False)

    data_simple.save_data("S2_Temp")
    data_ai.save_data("AI_Temp")
    data_simple.plot_data("S2_Temp")
    data_ai.plot_data("AI_Temp")

    print("Desired temperature: ", rooms_desired_temp)

    print("AI model data: ")
    print("Temperatures: (min, mean, max)")
    indoor_temp = np.array(data_ai.data['indoor_temp'])
    print(indoor_temp.min(), indoor_temp.mean(), indoor_temp.max())
    print("AI model standard deviation: ", np.std(data_ai.data['indoor_temp']))

    print("Simple two-state model data: ")
    print("Temperatures: (min, mean, max)")
    indoor_temp = np.array(data_simple.data['indoor_temp'])
    print(indoor_temp.min(), indoor_temp.mean(), indoor_temp.max())
    print("Simple model standard deviation: ", np.std(data_simple.data['indoor_temp']))


if __name__ == '__main__':
    run_silent_mode()
    sys.exit()
