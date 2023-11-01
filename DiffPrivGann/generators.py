import numpy as np
from numpy.random import normal as gaussgen

"""
Change of State Generator
"""


class DeltaStateGen:
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        initial_state: np.ndarray,
        gauss_parameters: np.ndarray,
    ):
        assert (
            gauss_parameters.shape[0] == B.shape[1]
        ), "guass_parameters must have the same number of rows as B has columns"
        assert (
            A.shape[1] == B.shape[0]
        ), "A must have the same number of columns as B has rows"

        self.A = A
        self.B = B
        self.C = C
        self.current_state = initial_state
        self.gauss_parameters = gauss_parameters

    def next_step(self) -> np.ndarray:
        mu_t = np.empty((self.B.shape[1], 1))
        for i, gauss_params in enumerate(self.gauss_parameters):
            mu = gauss_params[0]
            var = gauss_params[1]
            mu_t[i] = gaussgen(mu, var, 1)

        delta_x = self.A @ self.current_state + self.B @ mu_t
        self.current_state += delta_x

        return self.current_state

    def generate_observation(self, X):
        return self.C @ X


"""
We might want to delete the stuff below. Frankly we might not end up using it 
"""

"""
Below describes a Hidden Markov Model from which we will sample different 
Wide Sense Stationary Processes.

We can assume that one process is the normal funcitoning mode and the other are failure modes
"""


class HMMGenerator:
    def __init__(
        self,
        state_specifications: np.ndarray,
        state_probabilities: np.ndarray,
        initial_state: int,
    ):
        assert (
            state_probabilities.shape[0] == state_probabilities.shape[1]
        ), "State probabilities must be a square matrix"
        assert np.allclose(
            np.sum(state_probabilities, axis=1), np.ones(state_probabilities.shape[0])
        ), "State probabilities must sum to 1"

        self.state_probabilities = state_probabilities
        self.state_specifications = state_specifications
        self.current_state = initial_state

        # Create the Guassain Generators
        self.generators = []
        for i in range(self.state_specifications.shape[0]):
            self.generators.append(
                WSSGaussianGenerator(
                    dimensions=self.state_specifications.shape[1],
                    parameters=self.state_specifications[i, :],
                )
            )

    def sample(self):
        self.current_state = np.random.choice(
            self.state_specifications.shape[0], p=self.state_probabilities
        )
        draw_from_wss = self.generators[self.current_state].generate()
        return draw_from_wss


class WSSGaussianGenerator:
    def __init__(self, dimensions: int, parameters: np.ndarray):
        """
        args:
            parameters: (n,2) np.ndarray that serves as parameters for WSS Gaussian Process.
                0: mean
                1: variance
        """
        self.parameters = parameters

    def generate(self):
        samples = np.empty((self.parameters.shape[0], 1))
        for i, p in enumerate(self.parameters[0, :]):
            samples[i] = gaussgen()
