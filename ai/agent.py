import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from .a3c_model import A3CModel
from main import Environment as Env
from setup import ai


class Agent:
    EXP_COUNTER = 4000  # how many experiences (actions in environment) 1440 = day
    SAVE_DIR = './saves/'
    SAVE_FILE = 'a3c_model'
    BATCH_COUNT = 40

    @staticmethod
    def check_save_dir(save_dir=SAVE_DIR):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @staticmethod
    def save_model(model, save_file=SAVE_DIR + SAVE_FILE):
        model.save_weights(save_file)

    @staticmethod
    def load_model(model, load_file=SAVE_DIR + SAVE_FILE):
        if os.path.exists(load_file + '.index'):
            model.load_weights(load_file)
            print("Weights loaded successfully.")
        else:
            print("Weights file does not exist.")

    @staticmethod
    def choose_simulation_one_action(state, model, training=False):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

        action, _ = model(state_tensor, training=training)
        action = action[0, 0] > 0.5
        return action

    @staticmethod
    def choose_simulation_all_action(state, model, training=False):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)

        action, _ = model(state_tensor, training=training)
        action = tf.where(action > 0.5, 1, 0)
        return action

    @staticmethod
    def choose_action(state, model, training=False, epsilon=0.02):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)

        action, value = model(state_tensor, training=training)
        random_value = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)

        if random_value > epsilon:
            action = tf.where(action > 0.5, 1, 0)
        else:
            # last_action = tf.expand_dims(state_tensor[:, -1, 2], axis=1)
            # action = (last_action + action) / 3 * 2 > random_value

            random_value = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
            action = tf.cast(action < random_value, tf.int32)

        return action.numpy(), value.numpy()

    @staticmethod
    def unpack_exp_and_step(model, experiences, epoch=0):
        states, actions, advantages, rewards, next_val = zip(*experiences)

        actions = np.array(actions)
        advantages = np.array(advantages)
        rewards = np.array(rewards)
        next_val = np.array(next_val)
        states = np.array(states)

        actions = actions.reshape(-1, 1).astype(np.float32)
        advantages = advantages.reshape(-1, 1).astype(np.float32)
        rewards = rewards.reshape(-1, 1).astype(np.float32)
        next_val = next_val.reshape(-1, 1).astype(np.float32)

        if ai['DEBUG']:
            Agent.save_exp_to_csv(actions, advantages, rewards, next_val, epoch)

        states = states.reshape(-1, states.shape[-2], states.shape[-1])

        if ai['DEBUG']:
            Agent.save_states_to_csv(states, epoch)

        # create training batches
        dim = len(actions)
        shuffled_indices = np.random.permutation(dim)
        shuffled_actions = actions[shuffled_indices]
        shuffled_advantages = advantages[shuffled_indices]
        shuffled_rewards = rewards[shuffled_indices]
        shuffled_next_val = next_val[shuffled_indices]
        shuffled_states = states[shuffled_indices]
        split_actions = np.array_split(shuffled_actions, Agent.BATCH_COUNT)
        split_advantages = np.array_split(shuffled_advantages, Agent.BATCH_COUNT)
        split_rewards = np.array_split(shuffled_rewards, Agent.BATCH_COUNT)
        split_next_val = np.array_split(shuffled_next_val, Agent.BATCH_COUNT)
        split_states = np.array_split(shuffled_states, Agent.BATCH_COUNT)

        actor_loss, critic_loss, total_loss = [], [], []
        for env_state, actions, advantages, rewards, next_val in zip(split_states, split_actions, split_advantages,
                                                                    split_rewards, split_next_val):
            # ##################################
            # action_probs, values = model.call(env_state)
            # action_probs = tf.clip_by_value(action_probs, 1e-8, 1 - 1e-8)
            #
            # log_probs = tf.math.log(action_probs)
            # log_probs_neg = tf.math.log(1 - action_probs)
            #
            # selected_log_probs = actions * log_probs + (1 - actions) * log_probs_neg
            #
            # entropy = -(action_probs * log_probs + (1 - action_probs) * log_probs_neg)
            # mean_entropy = tf.reduce_mean(entropy)
            #
            # policy_loss = -tf.reduce_mean(selected_log_probs * advantages)  # minus for maximization
            # loss = policy_loss + 0.98 * mean_entropy
            # ###################################

            a, c, t = model.train_step(env_state, actions, advantages, rewards, next_val, epoch)
            actor_loss.append(a)
            critic_loss.append(c)
            total_loss.append(t)
        return np.mean(actor_loss), np.mean(critic_loss), np.mean(total_loss)

    @staticmethod
    def learn(agent_id, model_weights_queue, experience_queue, desired_temps, gamma=0.98):
        model = A3CModel()
        env = Env(desired_temps)
        states = env.reset()
        # lazy build
        model(tf.convert_to_tensor(states, dtype=tf.float32))
        new_weights = model_weights_queue.get(timeout=60)
        model.set_weights(new_weights)

        episode = 0
        while episode < Agent.EXP_COUNTER:
            actions, values = Agent.choose_action(states, model, True)
            next_states, rewards = env.step(actions, 1)  # one (1) or rebuild environment step()
            _, next_values = Agent.choose_action(next_states, model, True)

            rewards = rewards.reshape((-1, 1))
            target_value = rewards + next_values * gamma
            advantages = target_value - values

            experience = (states, actions, advantages, rewards, next_values)
            experience_queue.put(experience)  # ((agent_id, experience))
            states = next_states
            episode += 1

        tf.keras.backend.clear_session()

    @staticmethod
    def save_states_to_csv(states, epoch, output_dir='data'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        states = states.reshape(-1, states.shape[-1])
        exp_df = pd.DataFrame(states)
        file = os.path.join(output_dir, f'states{epoch}.csv')
        # save to file
        exp_df.to_csv(file, index=False)

    @staticmethod
    def save_exp_to_csv(actions, advantages, rewards, next_val, epoch, output_dir='data'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        actions = actions.reshape(-1).astype(np.float32)
        advantages = advantages.reshape(-1).astype(np.float32)
        rewards = rewards.reshape(-1).astype(np.float32)
        next_val = next_val.reshape(-1).astype(np.float32)

        exp_df = pd.DataFrame({
            'Action': actions,
            'Advantages': advantages,
            'Rewards': rewards,
            'Next Value': next_val
        })
        file = os.path.join(output_dir, f'exp{epoch}.csv')
        # save to file
        exp_df.to_csv(file, index=False)

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
