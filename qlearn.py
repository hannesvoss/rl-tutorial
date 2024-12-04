'''
In diesem Beispiel wird der Q-Learning-Algorithmus implementiert, um die optimale Policy für das Gridworld-Problem zu lernen.
Der Q-Learning-Algorithmus ist ein Off-Policy-Algorithmus, der die Q-Werte für eine Zielpolicy schätzt, während eine andere Verhaltenspolicy verwendet wird.
Off-Policy-Algorithmen lernen die Q-Werte für eine andere Policy als die, die sie tatsächlich ausführen, während On-Policy-Algorithmen die Q-Werte für die Policy schätzen, die sie ausführen.
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
    while state != terminal_state:
        # Aktion basierend auf Verhaltenspolicy auswählen
        action = epsilon_greedy_policy(state)
        
        # Übergang in der Umgebung
        next_state, reward = step(state, action)
        
        # Q-Learning Update (Off-Policy Update-Regel)
        best_next_action = np.argmax(Q[next_state])  # Beste Aktion gemäß Zielpolicy
        Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
        
        state = next_state  # Weiter zum nächsten Zustand

# Zeige die finale Q-Tabelle
print("Q-Tabelle nach Training:")
print(Q)