"""
@author: Robert

Tools for generating sequences from a probabilistic model
"""

import itertools
import numpy as np
import random


class SequenceSampler(object):
    """Object for generating sequences from a probabilistic model"""

    def gen_sequence(self):
        """Generate sequences"""
        self.reset()
        sequence = list(
            itertools.takewhile(lambda x: x != self.end_token, self.gen_sample())
        )
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

    def __init__(self, alpha, gamma, beta=1):
        """Initialize model with predefined parameters

        args:
            alpha: np.array. Transition probabilities.
            gamma: np.array. Initial value probabilities.
            beta: float. Decay constant.
        """
        assert alpha.ndim == 2, "ERROR: alpha not a matrix"
        assert alpha.shape[0] == alpha.shape[1] - 1, "ERROR: alpha not square"
        assert alpha.shape[0] == gamma.shape[0], "ERROR: incompatible alpha and gamma"
        assert 0 <= beta, "ERROR: beta must be non-negative"

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.end_token = alpha.shape[0]
        self.prev = None

    @classmethod
    def random_init(cls, n, zero_diag=True, beta=1):
        """Randomly initialize model parameters

        args:
            n: int. Number of states.
            zero_diag: bool. True if diagonal elements of transition matrix are
                forced to be zero (e.g. states do not repeat)
            beta: float. Decay constant.
        """
        gamma = np.random.rand(n)
        gamma[n - 1] = 0
        gamma = gamma / np.sum(gamma)

        alpha = np.random.rand(n, n)
        if zero_diag:
            np.fill_diagonal(alpha, 0)
        z = np.sum(alpha, axis=1).reshape((n, 1))
        alpha = alpha / z

        return cls(alpha, gamma)

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
        # BEGIN STUPID HACK
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


def transition_matrix(seqs, vocab, k=0):
    """Learn global Markov transition matrix from sequences

    Args:
        seqs: Contains sequences to learn from.
        vocab: Words in sequences.
        k: Smoothing parameter from Dirchlet prior.

    Returns:
        T: Transition matrix in the form of an np.array.
    """
    n = len(vocab)
    alpha = np.zeros((n, n + 1))  # Note: +1 for end_token
    gamma = np.zeros(n)
    # Fill with counts
    for seq in seqs:
        if len(seq) > 1:
            for i, j in zip(seq[:-1], seq[1:]):
                alpha[i, j] = alpha[i, j] + 1
            alpha[j, n] = alpha[j, n] + 1
        else:
            alpha[seq[0], n] = alpha[seq[0], n] + 1
        gamma[seq[0]] = gamma[seq[0]] + 1
    # Normalize
    z = np.sum(alpha, axis=1).reshape((n, 1))
    alpha = (alpha + k) / (z + n * k)
    gamma = gamma / np.sum(gamma)
    return alpha, gamma


if __name__ == '__main__':
    seqs = [[0, 1], [1, 0]]
    vocab = {0: 0, 1: 1}
    alpha, gamma = transition_matrix(seqs, vocab)
    print alpha
