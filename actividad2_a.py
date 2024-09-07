import numpy as np
import matplotlib.pyplot as plt

transition_matrix = np.array([
    [0.25, 0.06, 0.08, 0.15, 0.04, 0.02, 0.15, 0.15, 0.10],  # G1
    [0.15, 0.15, 0.10, 0.22, 0.01, 0.02, 0.15, 0.10, 0.10],  # G2
    [0.12, 0.00, 0.05, 0.24, 0.14, 0.04, 0.27, 0.07, 0.07],  # G3
    [0.05, 0.13, 0.05, 0.30, 0.10, 0.10, 0.22, 0.05, 0.00],  # G4
    [0.18, 0.20, 0.07, 0.20, 0.15, 0.05, 0.05, 0.05, 0.05],  # G5
    [0.20, 0.10, 0.20, 0.05, 0.05, 0.10, 0.02, 0.15, 0.13],  # G6
    [0.01, 0.05, 0.15, 0.14, 0.17, 0.10, 0.12, 0.10, 0.16],  # G7
    [0.17, 0.15, 0.07, 0.07, 0.15, 0.10, 0.12, 0.09, 0.08],  # G8
    [0.13, 0.11, 0.13, 0.03, 0.20, 0.20, 0.04, 0.15, 0.01]   # G9
])

groups = ['Helloween', 'Hammerfall', 'Stratovarius', 'Rhapsody of Fire', 'Yngwie Malmsteen',
          'Liquid Tension Experiment', 'Blind Guardian', 'Dream Theater', 'Symphony X']

# Parámetros de simulación
epsilon = 0.001  # Umbral de convergencia
max_iterations = 1000  # Máximo de iteraciones

initial_state = np.zeros(9)
initial_state[4] = 1  # G5

probabilities_history = [initial_state]

# Random walk hasta la convergencia
for _ in range(max_iterations):
    new_state = np.dot(probabilities_history[-1], transition_matrix)  # Multiplicación de matrices
    probabilities_history.append(new_state)
    
    # Criterio de convergencia
    if np.linalg.norm(new_state - probabilities_history[-2]) < epsilon:
        break

# Convertir la historia de probabilidades en un array para graficar
probabilities_history = np.array(probabilities_history)

for i in range(len(groups)):
    plt.plot(probabilities_history[:, i], label=groups[i])

plt.title("Evolución de probabilidades de escuchar canciones de los grupos")
plt.xlabel("Iteraciones")
plt.ylabel("Probabilidad")
plt.legend(loc='best')
plt.grid(True)
plt.show()

# probabilidades a las que converge
final_probabilities = probabilities_history[-1]
print("Probabilidades finales de convergencia:")
for i, group in enumerate(groups):
    print(f"{group}: {final_probabilities[i]:.4f}")
