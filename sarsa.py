'''
In diesem Beispiel wird der SARSA-Algorithmus (State-Action-Reward-State-Action) implementiert, um die optimale Policy für das Gridworld-Problem zu lernen.
Die Q-Tabelle wird iterativ durch Interaktion mit der Umgebung aktualisiert. Der SARSA-Algorithmus ist ein On-Policy-Algorithmus, der die Q-Werte für die aktuelle Policy schätzt.
On-Policy-Algorithmen lernen die Q-Werte für die Policy, die sie tatsächlich ausführen, während Off-Policy-Algorithmen die Q-Werte für eine andere Policy schätzen können.
'''

import numpy as np

# Umgebung
states = [0, 1, 2, 3, 4]  # Zustände
actions = [0, 1]  # 0 = links, 1 = rechts
gamma = 0.9  # Diskontierungsfaktor
alpha = 0.1  # Lernrate
epsilon = 0.1  # Epsilon-Greedy für Exploration

# Belohnungen und Übergänge
rewards = {4: 1}  # Terminalzustand gibt Belohnung von 1
terminal_state = 4

# Q-Tabelle initialisieren
Q = np.zeros((len(states), len(actions)))

# Schritt in der Umgebung ausführen
def step(state, action):
    if action == 0:  # Links
        next_state = max(0, state - 1)
    else:  # Rechts
        next_state = min(4, state + 1)
    reward = rewards.get(next_state, 0)  # Belohnung, falls vorhanden
    return next_state, reward

# Epsilon-greedy Policy
def epsilon_greedy_policy(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)  # Zufällige Aktion (Exploration)
    else:
        return np.argmax(Q[state])  # Aktion mit höchstem Q-Wert (Exploitation)

# Training
num_episodes = 1000

for episode in range(num_episodes):
    state = 0  # Startzustand
    action = epsilon_greedy_policy(state)  # Wähle erste Aktion gemäß der Policy
    
    while state != terminal_state:
        # Übergang in der Umgebung
        next_state, reward = step(state, action)
        next_action = epsilon_greedy_policy(next_state)  # Wähle nächste Aktion gemäß der Policy
        
        # SARSA Update (On-Policy Update-Regel)
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        
        # Übergang zum nächsten Zustand und Aktion
        state, action = next_state, next_action

# Zeige die finale Q-Tabelle
print("Q-Tabelle nach Training:")
print(Q)