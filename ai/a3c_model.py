import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, LeakyReLU, BatchNormalization, GlobalAveragePooling1D


class A3CModel(Model):
    LEARNING_RATE = 5.0e-04
    LEARNING_RATE_DECAY_FACTOR = 0.99
    CLIP_NORM = 1.0
    CLIP_NORM_RISE_FACTOR = 1.1

    def __init__(self, learning_rate=LEARNING_RATE):
        super(A3CModel, self).__init__()

        self.last_epoch = 0
        self.learning_rate = learning_rate
        self.clip_norm = self.CLIP_NORM

        # GRU Layer
        self.gru_one = GRU(128, return_sequences=True, return_state=False)
        self.gru_two = GRU(128, return_sequences=True, return_state=False)
        # self.gru_thr = GRU(64, return_sequences=True, return_state=False)
        # self.gru_out = GRU(128)

        self.ga_pool = GlobalAveragePooling1D()

        self.mid_dense = Dense(128)
        self.mid_activation = LeakyReLU(alpha=0.1)
        self.mid_norm = BatchNormalization()

        self.last_dense = Dense(64)
        self.last_activation = LeakyReLU(alpha=0.2)
        self.last_norm = BatchNormalization()

        # Actor-Critic output
        self.actor_dense = Dense(16)
        self.actor_activation = LeakyReLU(alpha=0.1)
        self.actor_norm = BatchNormalization()
        self.actor_out = Dense(1, activation='sigmoid')

        self.critic_dense = Dense(16)
        self.critic_activation = LeakyReLU(alpha=0.1)
        self.critic_norm = BatchNormalization()
        self.critic_out = Dense(1, activation='linear')

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        # Input Layer
        x = inputs

        x = self.gru_one(x)
        x = self.gru_two(x)
        # x = self.gru_thr(x)
        # x = self.gru_out(x)

        x = self.ga_pool(x)

        x = self.mid_dense(x)
        x = self.mid_norm(x)
        x = self.mid_activation(x)

        x = self.last_dense(x)
        x = self.last_norm(x)
        x = self.last_activation(x)

        a_out = self.actor_dense(x)
        a_out = self.actor_norm(a_out)
        a_out = self.actor_activation(a_out)

        c_out = self.critic_dense(x)
        c_out = self.critic_norm(c_out)
        c_out = self.critic_activation(c_out)

        actor_output = self.actor_out(a_out)
        critic_output = self.critic_out(c_out)
        return actor_output, critic_output

    def actor_loss(self, advantages, actions, action_probs, entropy_beta=0.01):
        action_probs = tf.clip_by_value(action_probs, 1e-8, 1 - 1e-8)

        log_probs = tf.math.log(action_probs)
        log_probs_neg = tf.math.log(1 - action_probs)

        selected_log_probs = actions * log_probs + (1 - actions) * log_probs_neg

        entropy = -(action_probs * log_probs + (1 - action_probs) * log_probs_neg)
        mean_entropy = tf.reduce_mean(entropy)

        #### TEST BOTH WERSION #####
        # Both got pros and cons

        # mean_advantages, variance_advantages = tf.nn.moments(selected_log_probs * advantages, axes=[0])
        # std_advantages = tf.sqrt(variance_advantages)
        #
        # policy_loss = tf.abs(mean_advantages) + std_advantages  # minus for maximization
        # loss = policy_loss + entropy_beta * mean_entropy

        #############################

        policy_loss = -tf.reduce_mean(selected_log_probs * advantages)  # minus for maximization
        loss = policy_loss + entropy_beta * mean_entropy

        return loss

    def critic_loss(self, true_values, estimated_values):
        return tf.keras.losses.mean_squared_error(true_values, estimated_values)

    @tf.function(reduce_retracing=True)
    def train_step(self, env_state, actions, advantages, rewards, next_values, epoch=0, gamma=0.98):
        if epoch != self.last_epoch:
            self.learning_rate = self.learning_rate * (self.LEARNING_RATE_DECAY_FACTOR ** epoch)
            self.optimizer.learning_rate.assign(self.learning_rate)
            self.clip_norm = self.clip_norm * self.CLIP_NORM_RISE_FACTOR
            self.last_epoch = epoch

        with tf.GradientTape() as tape:
            action_probs, values = self.call(env_state)

            actor_loss = self.actor_loss(advantages, actions, action_probs)

            true_values = rewards + gamma * next_values
            critic_loss = self.critic_loss(tf.squeeze(true_values), tf.squeeze(values))

            total_loss = tf.abs(actor_loss) + tf.abs(critic_loss)

        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return actor_loss, critic_loss, total_loss
