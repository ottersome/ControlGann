# %% Imports
import numpy as np
from numpy.random import normal as gaussgen

from DiffPrivGann import decoders, encoders, generators

# %% State Specifications
num_states = 3
A = gaussgen(0, 1, (num_states, num_states))
B = gaussgen(0, 1, (num_states, num_states))
C = np.array([[1, 2, 3], [0, 2, 3], [0, 0, 3]])

initial_state = np.array([[40], [85], [0.5]])
gauss_parameters = np.array([[0, 1], [0, 1], [0, 1]])

number_of_steps = 50

# %% Create the Environment
hmmChain = generators.DeltaStateGen(A, B, C, initial_state, gauss_parameters)

# %% Let this run for a while

x_global = np.empty((num_states, number_of_steps))
y_global = np.empty((num_states, number_of_steps))

for step in range(number_of_steps):
    x_global[:, step] = hmmChain.next_step().squeeze()
    y_global[:, step] = hmmChain.generate_observation(x_global[:, step])

# %% Graph how the state changes
import matplotlib.pyplot as plt

plt.style.use("rose-pine-dawn")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for state in range(num_states):
    ax[0].plot(x_global[state, :], label=f"State {state}")
    ax[0].set_title("State")
    ax[0].legend()

    ax[1].plot(y_global[state, :], label=f"Observation {state}")
    ax[1].set_title("Observation")
    ax[0].legend()

plt.tight_layout()
plt.show()
