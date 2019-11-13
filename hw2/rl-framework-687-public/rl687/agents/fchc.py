import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """
    
    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, numEpisodes:int=10):
        self._init_theta = theta
        self._init_sigma = sigma
        self._name = "First_Choice_Hill_Climbing"

        self._theta = theta
        self._Sigma = sigma
        self._numEpisodes = numEpisodes
        self._evaluationFunction = evaluationFunction

        # self._bestTheta = self._theta
        self._bestObject = self._evaluationFunction(self._theta, self._numEpisodes)

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._theta

    def train(self)->np.ndarray:
        # Sample policy
        ntheta = np.random.multivariate_normal(self._theta, self._Sigma * np.eye(self._theta.size))
        nobject = self._evaluationFunction(ntheta, self._numEpisodes)
        if nobject > self._bestObject:
            self._bestObject = nobject
            self._theta = ntheta
        return ntheta
    def reset(self)->None:
        self._theta = self._init_theta
