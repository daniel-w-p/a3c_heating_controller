import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, LeakyReLU, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import setup


class A3CModel(Model):
    LEARNING_RATE = 0.001
    CLIP_NORM = 50.0

    def __init__(self, state_size, action_size, learning_rate=LEARNING_RATE):
        super(A3CModel, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Input Layer
        self.input_layer = Input(shape=(self.state_size[0], self.state_size[1]))

        # GRU Layer
        self.gru = GRU(64, return_sequences=True, return_state=False)
        self.gru_out = GRU(32)

        # Actor-Critic output
        self.last_dense = Dense(64)
        self.last_activation = LeakyReLU(alpha=0.1)
        self.actor_out = Dense(self.action_size, activation='softmax')
        self.critic_out = Dense(1, activation='linear')

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        x = self.input_layer(inputs)
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

        selected_log_probs = tf.reduce_sum(log_probs * actions, axis=1, keepdims=True)

        entropy = -tf.reduce_sum(action_probs * log_probs, axis=1)
        mean_entropy = tf.reduce_mean(entropy)

        advantages = tf.squeeze(advantages, axis=1)
        policy_loss = -tf.reduce_mean(selected_log_probs * advantages)
        loss = policy_loss - entropy_beta * mean_entropy

        return loss

    def critic_loss(self, estimated_values, true_values):
        return tf.keras.losses.mean_squared_error(true_values, estimated_values)

    @tf.function(reduce_retracing=True)
    def train_step(self, env_state, plr_state, one_hot_action, advantages, rewards):

        with tf.GradientTape() as tape:
            action_probs, values = self.call((env_state, plr_state))

            actor_loss = self.actor_loss(advantages, one_hot_action, action_probs)

            critic_loss = self.critic_loss(rewards, tf.squeeze(values))

            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.CLIP_NORM)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return actor_loss, critic_loss, total_loss
