import sys
import pygame

from setup import gui, ai, AppMode
from main import Environment, TwoStateSwitch
from ai import A3CModel, Agent
from simulation import Simulator


COUNT_ROOMS = 4


def make_step(model, env, state):
    if ai['RUN_MODE'] == AppMode.COMPARE:
        actions = [model.choose_action(state) for _ in range(COUNT_ROOMS)]
    else:  # if ai['RUN_MODE'] == AppMode.RUN:
        actions = [Agent.choose_action(state[i], model, False, True)[0] for i in range(COUNT_ROOMS)]

    state, _ = env.step(actions, 1)  # 1 - one minute
    return state


def run_simulator():
    pygame.init()
    screen = pygame.display.set_mode((gui['SCREEN_WIDTH'], gui['SCREEN_HEIGHT']))
    pygame.display.set_caption(gui['WINDOW_TITLE'])
    font = pygame.font.Font(None, 24)
    start_time = pygame.time.get_ticks()

    # only for AI
    model = None
    state_shape = (20, 6)
    run_from_checkpoint = True

    # environment
    rooms_desired_temps = [19, 20, 21, 22]
    if len(rooms_desired_temps) != COUNT_ROOMS:
        print('Room numbers do not match!!!!')
    env = Environment(rooms_desired_temps)
    simulator = Simulator(screen, font)

    if ai['RUN_MODE'] == AppMode.COMPARE:
        print("Start app in COMPARE mode (simple controller)")
        model = TwoStateSwitch()
    elif ai['RUN_MODE'] == AppMode.RUN:
        print("Start app in RUN mode (A3C controller)")
        model = A3CModel()
        if run_from_checkpoint:
            Agent.load_model(model)

    simulator.run(model, env, callback=make_step)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        first_argument = sys.argv[1]
        if first_argument == 'c':  # compare
            ai['RUN_MODE'] = AppMode.COMPARE

    run_simulator()
    pygame.quit()
    sys.exit()
