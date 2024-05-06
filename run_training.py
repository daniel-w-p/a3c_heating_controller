import multiprocessing as mp
import os
import queue

import numpy as np
import tensorflow as tf

from ai import Agent, A3CModel
from main import Environment


def main():
    num_agents = 10
    epochs = 30
    start_from_checkpoint = True
    desired_temps = [17., 17.5, 18., 18.5, 19., 19.5, 20., 20.5, 21., 21.5]

    # Dynamic GPU memory allocation for TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    env = Environment(desired_temps)
    main_model = A3CModel()
    # Lazy build
    states = env.reset()
    # here some rooms [len(desired_temps)] are treated as a bach for single room
    main_model(tf.convert_to_tensor(states, dtype=tf.float32))

    main_model.summary()

    if start_from_checkpoint and os.listdir(Agent.SAVE_DIR):
        Agent.load_model(main_model)

    manager = mp.Manager()
    actor_losses = []
    critic_losses = []
    total_losses = []

    for i in range(epochs):
        print("Creating Agents")
        weights_queue = manager.Queue()
        experience_queue = manager.Queue()
        agents = []
        main_model_weights = main_model.get_weights()
        desired_temps = [17., 17.5, 18., 18.5, 19., 19.5, 20., 20.5, 21., 21.5]
        experiences = []

        # Prepare and run agents (multiprocessing)
        for a in range(num_agents):
            weights_queue.put(main_model_weights)
            desired_temps = np.array(desired_temps) + 0.13
            print("Creating Agent ", a)
            agent_process = mp.Process(target=Agent.learn,
                                       args=(a, weights_queue, experience_queue, desired_temps))
            agents.append(agent_process)
            agent_process.start()

        print(f"Starting training epoch: {i}")

        # For progress monitoring
        total_steps = Agent.EXP_COUNTER * num_agents

        while True:
            try:
                data = experience_queue.get(timeout=60)
                experiences.append(data)
            except queue.Empty:
                print("Empty queue")
            except EOFError:
                print("Queue read error")

            actual_step = len(experiences)

            if actual_step >= total_steps:
                print(f'\rEpoch: {i} --> 100% Complete ')
                print("Total experiences:", len(experiences))
                break  # when collect all

        # Fin
        for agent in agents:
            agent.join()

        # Some logs.
        print(f'Epoch {i} finished. Updating main model weights')
        rewards = [reward for _, _, _, reward, _ in experiences]
        print(f'Average reward: {np.mean(rewards)}')
        print(f'Max reward: {np.max(rewards)}')
        print(f'Min reward: {np.min(rewards)}')
        print(f'Rewards shape: {np.array(rewards).shape}')

        # Update the main model based on the experiences collected from agents.
        actor_loss, critic_loss, total_loss = Agent.unpack_exp_and_step(main_model, experiences, i)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        total_losses.append(total_loss)

        print(f"Actual learning rate: {main_model.learning_rate} in epoch {i}")
        print(f"Losses:\n t - {total_losses} ;\n a - {actor_losses} ;\n c - {critic_losses}")

        if i > 0 and i % 5 == 0:  # save interval - 5 epochs
            epoch_dir = f'epoch_{i}/'
            main_model.save_weights(Agent.SAVE_DIR+epoch_dir+Agent.SAVE_FILE)

    # Save last epoch in main localization
    main_model.save_weights(Agent.SAVE_DIR + Agent.SAVE_FILE)
    # Save losses
    Agent.save_losses_csv(actor_losses, critic_losses, total_losses)
    # Plot losses
    Agent.plot_losses(actor_losses, critic_losses, total_losses)


if __name__ == "__main__":
    print("This module is not fully implemented yet")
    mp.set_start_method('spawn')

    Agent.check_save_dir()

    main()

    print("Done!")
