'''
In diesem Beispiel wird der REINFORCE-Algorithmus implementiert, um die optimale Policy für das 1D-Bewegungsproblem zu lernen.
Der REINFORCE-Algorithmus ist ein Policy-Gradienten-Algorithmus, der die Policy direkt durch Gradientenabstieg im Parameterraum optimiert.
Im Gegensatz zu Q-Learning und SARSA lernt REINFORCE eine Policy, die die Aktionen direkt aus den Zuständen auswählt, anstatt Q-Werte zu schätzen.

Der REINFORCE-Algorithmus verwendet eine Softmax-Policy, um die Wahrscheinlichkeiten für die Aktionen in jedem Zustand zu berechnen.
Die Policy-Parameter werden durch Gradientenabstieg aktualisiert, wobei die Belohnungen der Episoden als Gewichtungsfaktoren verwendet werden.
'''

import numpy as np

# Umgebung: 1D-Bewegung
states = [0, 1, 2, 3, 4]  # Zustände
actions = [0, 1]  # 0 = links, 1 = rechts
gamma = 0.9  # Diskontierungsfaktor
alpha = 0.01  # Lernrate

# Policy-Parameter (z. B. für Softmax)
theta = np.random.rand(len(states), len(actions))

# Softmax-Policy
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def select_action(state):
    probs = softmax(theta[state])
    return np.random.choice(actions, p=probs)

# Umgebungsschritt
def step(state, action):
    if action == 0:  # Links
        next_state = max(0, state - 1)
    else:  # Rechts
        next_state = min(4, state + 1)
    reward = 1 if next_state == 4 else 0
    return next_state, reward

# Generiere eine Episode
def generate_episode():
    state = 0
    episode = []
    while state != 4:
        action = select_action(state)
        next_state, reward = step(state, action)
        episode.append((state, action, reward))
        state = next_state
    return episode

# Training
num_episodes = 1000

for episode in range(num_episodes):
    episode_data = generate_episode()
    G = 0
    for t in reversed(range(len(episode_data))):
        state, action, reward = episode_data[t]
        G = reward + gamma * G
        
        gradient = -softmax(theta[state])
        gradient[action] += 1
        
        theta[state] += alpha * gradient * G

# Zeige die trainierte Policy
print("Trainierte Policy (Wahrscheinlichkeiten):")
for s in states:
    print(f"State {s}: {softmax(theta[s])}")