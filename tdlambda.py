import numpy as np
import gym

# Umgebung initialisieren
env = gym.make("CliffWalking-v0")  # Beispiel mit diskreter Umgebung
state_dim = env.observation_space.n
action_dim = env.action_space.n

# Hyperparameter
alpha = 0.1  # Lernrate
gamma = 0.9  # Diskontierungsfaktor
_lambda = 0.8  # Lambda-Wert für TD(λ)
episodes = 500

# Wertfunktion initialisieren
V = np.zeros(state_dim)

# TD(λ) Algorithmus
for episode in range(episodes):
    state = env.reset()
    eligibility_trace = np.zeros(state_dim)  # Spuren initialisieren
    
    while True:
        # Aktion basierend auf einer Zufallspolitik auswählen
        action = env.action_space.sample()
        
        # Schritt in der Umgebung
        next_state, reward, done, _ = env.step(action)
        
        # TD-Fehler berechnen
        delta = reward + gamma * V[next_state] - V[state]
        
        # Eligibility Trace aktualisieren
        eligibility_trace[state] += 1  # Für besuchten Zustand erhöhen
        
        # Wertefunktion aktualisieren
        V += alpha * delta * eligibility_trace
        
        # Spuren für alle Zustände mit \(\lambda\) und \(\gamma\) reduzieren
        eligibility_trace *= gamma * _lambda
        
        # Transition zum nächsten Zustand
        state = next_state
        
        # Episode beenden, wenn Terminalzustand erreicht ist
        if done:
            break

# Ergebnisse ausgeben
print("Finale Wertefunktion:")
print(V)
