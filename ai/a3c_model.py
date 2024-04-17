import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, LeakyReLU

import setup


class A3CModel(Model):
    LEARNING_RATE = 0.001
    CLIP_NORM = 50.0

    def __init__(self, learning_rate=LEARNING_RATE):
        super(A3CModel, self).__init__()

        self.learning_rate = learning_rate

        # GRU Layer
        self.gru = GRU(64, return_sequences=True, return_state=False)
        self.gru_out = GRU(32)

        # Actor-Critic output
        self.last_dense = Dense(64)
        self.last_activation = LeakyReLU(alpha=0.1)
        self.actor_out = Dense(1, activation='sigmoid')
        self.critic_out = Dense(1, activation='linear')

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        # Input Layer
        x = inputs

        x = self.gru(x)
        x = self.gru_out(x)
        x = self.last_dense(x)
        x = self.last_activation(x)
        actor_output = self.actor_out(x)
        critic_output = self.critic_out(x)
        return actor_output, critic_output

    def actor_loss(self, advantages, actions, action_probs, entropy_beta=0.01):
        action_probs = tf.clip_by_value(action_probs, 1e-8, 1 - 1e-8)

        log_probs = tf.math.log(action_probs)
        log_probs_neg = tf.math.log(1 - action_probs)

        selected_log_probs = actions * log_probs + (1 - actions) * log_probs_neg

        entropy = -(action_probs * log_probs + (1 - action_probs) * log_probs_neg)
        mean_entropy = tf.reduce_mean(entropy)

        policy_loss = -tf.reduce_mean(selected_log_probs * advantages)  # minus for maximization
        loss = policy_loss + entropy_beta * mean_entropy

        return loss

    def critic_loss(self, true_values, estimated_values):
        return tf.keras.losses.mean_squared_error(true_values, estimated_values)

    @tf.function(reduce_retracing=True)
    def train_step(self, env_state, action, advantages, rewards, next_values, gamma=0.98):
        with tf.GradientTape() as tape:
            action_probs, values = self.call(env_state)

            actor_loss = self.actor_loss(advantages, action, action_probs)

            true_values = rewards + gamma * tf.squeeze(next_values)
            critic_loss = self.critic_loss(true_values, tf.squeeze(values))

            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.CLIP_NORM)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return actor_loss, critic_loss, total_loss
