import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from .a3c_model import A3CModel
from main import Environment as Env


class Agent:
    EXP_COUNTER = 1440  # how many experiences (actions in environment) 1440 = day
    SAVE_DIR = './saves/'
    SAVE_FILE = 'a3c_model'

    @staticmethod
    def check_save_dir(save_dir=SAVE_DIR):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @staticmethod
    def save_model(model, save_file=SAVE_DIR+SAVE_FILE):
        model.save_weights(save_file)

    @staticmethod
    def load_model(model, load_file=SAVE_DIR+SAVE_FILE):
        if os.path.exists(load_file+'.index'):
            model.load_weights(load_file)
            print("Weights loaded successfully.")
        else:
            print("Weights file does not exist.")

    @staticmethod
    def choose_action(state, model, training=False, simulate=False, epsilon=0.1):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

        action, value = model(state_tensor, training=training)

        if simulate:
            action = tf.where(action < 0.5, 0, 1)
        else:
            random_value = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
            random_choice = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)  # choose 0 lub 1
            action = tf.cond(
                random_value < epsilon,
                true_fn=lambda: random_choice,
                false_fn=lambda: tf.cast(action[0, 0] < random_value, tf.int32)
            )

        return action.numpy(), value[0, 0].numpy()

    @staticmethod
    def unpack_exp_and_step(model, experiences):
        states, actions, advantages, rewards = zip(*experiences)

        actions = np.array(actions)
        advantages = np.array(advantages)
        rewards = np.array(rewards)
        states = np.array(states)

        actions = actions.reshape(-1).astype(np.float32)
        advantages = advantages.reshape(-1).astype(np.float32)
        rewards = rewards.reshape(-1).astype(np.float32)

        states = states.reshape(-1, states.shape[-2], states.shape[-1])

        action = tf.convert_to_tensor(actions, dtype=tf.float32)
        env_state = tf.convert_to_tensor(states, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        return model.train_step(env_state, action, advantages, rewards)

    @staticmethod
    def learn(agent_id, model_weights_queue, experience_queue, desired_temps, gamma=0.98):
        count_rooms = len(desired_temps)
        model = A3CModel()
        env = Env(desired_temps)
        states = env.reset()
        # lazy build
        model(tf.convert_to_tensor(states, dtype=tf.float32))
        new_weights = model_weights_queue.get(timeout=60)
        model.set_weights(new_weights)

        episode = 0
        local_experience = []
        while episode < Agent.EXP_COUNTER:
            actions, values = zip(*[Agent.choose_action(states[i], model, True) for i in range(count_rooms)])
            next_states, rewards = env.step(actions, episode)
            _, next_values = zip(*[Agent.choose_action(next_states[i], model, True) for i in range(count_rooms)])

            target_value = np.array(rewards) + np.array(next_values) * gamma
            advantages = target_value - np.array(values)

            experience = (states, actions, advantages, rewards)
            experience_queue.put(experience)  # ((agent_id, experience))
            local_experience.append(experience)
            episode += 1

            if episode % 101 == 0:
                Agent.unpack_exp_and_step(model, local_experience)
                local_experience.clear()

        tf.keras.backend.clear_session()

    # TODO work on those function
    @staticmethod
    def save_losses_csv(actor_losses, critic_losses, total_losses, output_dir='data/losses'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        losses_df = pd.DataFrame({
            'Actor Loss': actor_losses,
            'Critic Loss': critic_losses,
            'Total Loss': total_losses
        })
        file = os.path.join(output_dir, 'losses.csv')
        # save to file
        losses_df.to_csv(file, index=False)

    @staticmethod
    def plot_losses(actor_losses, critic_losses, total_losses, output_dir='data/losses'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = os.path.join(output_dir, 'losses.png')

        plt.figure(figsize=(5, 5))
        plt.plot(actor_losses, label='Actor Loss')
        plt.plot(critic_losses, label='Critic Loss')
        plt.plot(total_losses, label='Total Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Losses over Time')
        plt.legend()
        plt.savefig(file)
        plt.close()

    @staticmethod
    def visualize_feature_maps(model, input_map, output_dir='data/feature_maps'):

        feature_maps = model.map_nn.predict(input_map)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(feature_maps.shape[-1]):
            plt.figure(figsize=(2, 2))
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')
            plt.savefig(f'{output_dir}/feature_map_{i}.png')
            plt.close()

