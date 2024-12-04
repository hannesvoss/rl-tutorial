import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
import random
from collections import deque

# Umgebung initialisieren
env = gym.make("Pendulum-v1")  # Beispiel mit kontinuierlichem Aktionsraum
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# Replay-Buffer
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# Actor-Netzwerk
def create_actor():
    inputs = layers.Input(shape=(state_dim,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(action_dim, activation="tanh")(out) * action_bound
    model = tf.keras.Model(inputs, outputs)
    return model

# Critic-Netzwerk
def create_critic():
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    concat = layers.Concatenate()([state_input, action_input])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

# Update Target Networks
def update_target_weights(target_model, model, tau):
    for target_param, param in zip(target_model.trainable_weights, model.trainable_weights):
        target_param.assign(tau * param + (1 - tau) * target_param)

# Hyperparameter
gamma = 0.99
tau = 0.005
actor_lr = 0.001
critic_lr = 0.002
buffer = ReplayBuffer()
batch_size = 64

# Netzwerke initialisieren
actor = create_actor()
critic = create_critic()
target_actor = create_actor()
target_critic = create_critic()
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

# Optimizer
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

# Training
num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(200):
        # Aktion mit Exploration (Ornstein-Uhlenbeck Rauschen)
        action = actor(np.expand_dims(state, axis=0)).numpy()[0]
        noise = np.random.normal(0, 0.1, size=action_dim)
        action = np.clip(action + noise, -action_bound, action_bound)

        # Schritt in der Umgebung
        next_state, reward, done, _ = env.step(action)
        buffer.add((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        # Training, wenn genügend Daten im Replay-Buffer sind
        if len(buffer.buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # Critic-Update
            with tf.GradientTape() as tape:
                target_actions = target_actor(next_states)
                target_q = target_critic([next_states, target_actions])
                y = rewards + gamma * (1 - dones) * tf.squeeze(target_q)
                q_values = tf.squeeze(critic([states, actions]))
                critic_loss = tf.reduce_mean(tf.square(y - q_values))
            grads = tape.gradient(critic_loss, critic.trainable_weights)
            critic_optimizer.apply_gradients(zip(grads, critic.trainable_weights))

            # Actor-Update
            with tf.GradientTape() as tape:
                actions_pred = actor(states)
                actor_loss = -tf.reduce_mean(critic([states, actions_pred]))
            grads = tape.gradient(actor_loss, actor.trainable_weights)
            actor_optimizer.apply_gradients(zip(grads, actor.trainable_weights))

            # Update Target Networks
            update_target_weights(target_actor, actor, tau)
            update_target_weights(target_critic, critic, tau)

    print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}")

env.close()
