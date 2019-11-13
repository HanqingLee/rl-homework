import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean theta and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta: np.ndarray, sigma: float, popSize: int, numElite: int, numEpisodes: int,
                 evaluationFunction: Callable, epsilon: float = 1.5):
        self._init_theta = theta
        self._init_sigma = sigma
        self._name = "Cross_Entropy_Method"

        self._theta = theta
        self._Sigma = np.eye(theta.size) * sigma
        self._popSize = popSize
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._evaluationFunction = evaluationFunction
        self._epsilon = epsilon

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._theta

    def train(self) -> np.ndarray:
        sample_list = []
        for k in range(self._popSize):
            # Sample policy
            theta_k = np.random.multivariate_normal(self._theta, self._Sigma)
            # Evaluate
            object_k = self._evaluationFunction(theta_k, self._numEpisodes)
            sample_list.append([theta_k, object_k])
        sample_list.sort(key=lambda sample: sample[1], reverse=True)
        elite_sample = sample_list[:self._numElite]
        elite_theta = np.array([sample[0] for sample in elite_sample])
        ntheta = np.mean(elite_theta, axis=0)
        theta_diff = elite_theta - ntheta
        nsigma = (self._epsilon * np.eye(self._theta.size) + np.sum(np.einsum('ij,ki->ijk', theta_diff, theta_diff.T),
                                                                    axis=0)) / (
                         self._epsilon + self._numElite)
        # Update
        self._theta = ntheta
        self._Sigma = nsigma
        return ntheta

    def reset(self) -> None:
        self._theta = self._init_theta
        self._Sigma = np.eye(self._init_theta.size) * self._init_sigma
