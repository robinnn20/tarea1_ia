import numpy as np

# Definir la matriz de transición
transition_matrix = np.array([
    [0.25, 0.06, 0.08, 0.15, 0.04, 0.02, 0.15, 0.15, 0.10],
    [0.15, 0.15, 0.10, 0.22, 0.01, 0.02, 0.15, 0.10, 0.10],
    [0.12, 0.00, 0.05, 0.24, 0.14, 0.04, 0.27, 0.07, 0.07],
    [0.05, 0.13, 0.05, 0.30, 0.10, 0.10, 0.22, 0.05, 0.00],
    [0.18, 0.20, 0.07, 0.20, 0.15, 0.05, 0.05, 0.05, 0.05],
    [0.20, 0.10, 0.20, 0.05, 0.05, 0.10, 0.02, 0.15, 0.13],
    [0.01, 0.05, 0.15, 0.14, 0.17, 0.10, 0.12, 0.10, 0.16],
    [0.17, 0.15, 0.07, 0.07, 0.15, 0.10, 0.12, 0.09, 0.08],
    [0.13, 0.11, 0.13, 0.03, 0.20, 0.20, 0.04, 0.15, 0.01]
])

# Número de estados
num_states = transition_matrix.shape[0]

# Formar el sistema de ecuaciones
# π P = π
# π (P - I) = 0
# Agregar la condición de normalización: sum(π) = 1

A = np.transpose(transition_matrix) - np.identity(num_states)
A = np.vstack([A, np.ones(num_states)])

b = np.zeros(num_states)
b = np.append(b, 1)

pi = np.linalg.lstsq(A, b, rcond=None)[0]

print("Estados Estacionarios:")
for i in range(num_states):
    print(f"G{i+1}: {pi[i]:.4f}")
