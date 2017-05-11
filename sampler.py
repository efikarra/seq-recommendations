"""
@author: Robert

Tools for generating sequences from a probabilistic model
"""

import itertools
import math
import numpy as np
import random

class SequenceSampler(object):
    """Object for generating sequences from a probabilistic model"""

    def gen_sequence(self, n=None):
        """Generate sequences

        args:
            n: int. If not using end token, then this is the length of the
                sequence to return.

        """
        self.reset()
        if self.use_end_token:
            sequence = list(
                itertools.takewhile(lambda x: x != self.end_token, self.gen_sample())
            )
        else:
            assert n > 0, "ERROR: Must specify sequence length if not using an end token"
            generator = self.gen_sample()
            sequence = list(next(generator) for _ in xrange(n))
        return sequence

    def gen_sample(self):
        """Generate a sample"""
        raise NotImplementedError

    def reset(self):
        """Reset internal state"""
        pass


class MCSampler(SequenceSampler):
    """Generate sequences from a first-order Markov model

    Note: n-th state always considered to be an end state.
    """

    def __init__(self, alpha, gamma, beta=1, use_end_token=True):
        """Initialize model with predefined parameters

        args:
            alpha: np.array. Transition probabilities.
            gamma: np.array. Initial value probabilities.
            beta: float. Decay constant.
            use_end_token: bool. If True then an end token is used to stop
                sequence generation.
        """
        assert alpha.ndim == 2, "ERROR: alpha not a matrix"
        if use_end_token:
            assert alpha.shape[0] == alpha.shape[1] - 1, "ERROR: alpha not square"
        else:
            assert alpha.shape[0] == alpha.shape[1], "ERROR: alpha not square"
        assert alpha.shape[0] == gamma.shape[0], "ERROR: incompatible alpha and gamma"
        assert 0 <= beta, "ERROR: beta must be non-negative"

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.use_end_token = use_end_token

        if use_end_token:
            self.end_token = alpha.shape[0]
        else:
            self.end_token = None

        self.prev = None

    @classmethod
    def random_init(cls, n, zero_diag=True, **kwargs):
        """Randomly initialize model parameters

        args:
            n: int. Number of states.
            zero_diag: bool. True if diagonal elements of transition matrix are
                forced to be zero (e.g. states do not repeat)

            see __init__ method for other args.
        """
        gamma = np.random.rand(n)
        gamma[n - 1] = 0
        gamma = gamma / np.sum(gamma)

        alpha = np.random.rand(n, n)
        if zero_diag:
            np.fill_diagonal(alpha, 0)
        z = np.sum(alpha, axis=1).reshape((n, 1))
        alpha = alpha / z

        return cls(alpha, gamma, **kwargs)

    def gen_sample(self):
        while True:
            if self.prev is None:
                prob = self.gamma
                # Need to copy alpha since it mutates with decay
                alpha = np.copy(self.alpha)
            else:
                prob = alpha[self.prev]
            cdf = np.cumsum(prob)
            rng = random.random()
            sample = np.argmax(cdf > rng)
            if sample != self.end_token:
                self.prev = sample
                alpha = self.decay(alpha, sample)
            yield sample

    def decay(self, alpha, sample):
        alpha[:, sample] = self.beta * alpha[:, sample]
        z = np.sum(alpha, axis=1).reshape((alpha.shape[0], 1))
        # BEGIN STUPID HACK - NOTE: This will cause issues if not using an end
        # token
        bad_inds = z == 0.
        if np.sum(bad_inds) > 0:
            bad_inds = bad_inds.reshape((alpha.shape[0]))
            repl = np.zeros((1, alpha.shape[1]))
            repl[0, self.end_token] = 1
            alpha[bad_inds] = repl
            z[bad_inds] = 1
        # END STUPID HACK
        return alpha / z

    def reset(self):
        self.prev = None


# NOTE: RandomWalkSampler currently operates under the assumption that the
# transition probabilities to neighbors are uniform.
class RandomWalkSampler(SequenceSampler):
    """Generate sequences from a random walk with reinforcement on a grid"""

    def __init__(self, k, betas):
        """Initialize model

        args:
            k: Board size
            betas: Values that beta can be drawn from
        """
        self.k = k
        self.pos = (np.random.randint(0, self.k), np.random.randint(0, self.k))
        self.x = np.zeros((self.k, self.k))
        self.beta = np.random.choice(betas, size=(self.k, self.k))
        self.use_end_token = False # TODO: Refactor to avoid this stupid hack

    def gen_sample(self):
        while True:
            u, v = self.pos # Initial location
            self.x[u][v] += 1 # Note that agent has visited

            # Calculate local pdf
            neighbors = self.get_neighbors()
            pdf = np.array([self.beta[s][t]**self.x[s][t] for s, t in neighbors])
            pdf = pdf / np.sum(pdf)
            cdf = np.cumsum(pdf)

            # Move to new location
            rng = random.random()
            sample = np.argmax(cdf > rng)
            self.pos = neighbors[sample]

            yield self.pos

    def get_neighbors(self):
        """Get coordinates of adjacent cells"""
        u, v = self.pos
        h_range = [u-1, u, u+1]
        v_range = [v-1, v, v+1]
        neighbors = [pos for pos in itertools.product(h_range, v_range)
                   if (pos != self.pos) and (self._on_board(pos))]
        return neighbors

    def _on_board(self, pos):
        """Returns True if pos has valid coordinates, False otherwise"""
        h_check = (0 <= pos[0] < self.k)
        v_check = (0 <= pos[1] < self.k)
        if h_check and v_check:
            return True
        else:
            return False

    def reset(self):
        # Resets player position and visit counts. Betas kept the same.
        self.pos = (np.random.randint(0, self.k), np.random.randint(0, self.k))
        self.x = np.zeros((self.k, self.k))


if __name__ == '__main__':
    from utils import random_walk_animation

    # Example random walk
    rw_sampler = RandomWalkSampler(10, 'negative')
    random_walk_animation(rw_sampler, 'test.mp4')

