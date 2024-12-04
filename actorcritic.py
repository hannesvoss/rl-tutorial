import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Umgebung initialisieren
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Hyperparameter
gamma = 0.99
learning_rate = 0.001
episodes = 500
max_steps = 200

# Actor-Netzwerk
def create_actor():
    inputs = layers.Input(shape=(state_dim,))
    out = layers.Dense(24, activation="relu")(inputs)
    out = layers.Dense(24, activation="relu")(out)
    outputs = layers.Dense(action_dim, activation="softmax")(out)
    return tf.keras.Model(inputs, outputs)

# Critic-Netzwerk
def create_critic():
    inputs = layers.Input(shape=(state_dim,))
    out = layers.Dense(24, activation="relu")(inputs)
    out = layers.Dense(24, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    return tf.keras.Model(inputs, outputs)

# Netzwerke initialisieren
actor = create_actor()
critic = create_critic()

# Optimizer
actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

# Training
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_dim])
    episode_reward = 0

    for step in range(max_steps):
        # Aktion auswählen (stochastisch)
        probs = actor(state).numpy()[0]
        action = np.random.choice(action_dim, p=probs)

        # Schritt in der Umgebung
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_dim])

        # Zielwert berechnen
        value = critic(state).numpy()[0]
        next_value = critic(next_state).numpy()[0]
        target = reward + (1 - done) * gamma * next_value
        delta = target - value  # Advantage

        # Critic-Update
        with tf.GradientTape() as tape:
            value_pred = critic(state)
            critic_loss = tf.reduce_mean(tf.square(target - value_pred))
        grads = tape.gradient(critic_loss, critic.trainable_weights)
        critic_optimizer.apply_gradients(zip(grads, critic.trainable_weights))

        # Actor-Update
        with tf.GradientTape() as tape:
            action_probs = actor(state)
            log_prob = tf.math.log(action_probs[0, action])
            actor_loss = -log_prob * delta  # Policy Gradient mit Advantage
        grads = tape.gradient(actor_loss, actor.trainable_weights)
        actor_optimizer.apply_gradients(zip(grads, actor.trainable_weights))

        # Übergang zum nächsten Zustand
        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}: Reward: {episode_reward}")

env.close()