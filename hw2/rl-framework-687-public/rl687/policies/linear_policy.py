import numpy as np
from .skeleton import Policy
from typing import Union

class LinearSoftmax(Policy):
    def __init__(self, degree:int, numActions: int, state_size: int ):
        # The internal policy parameters must be stored as a matrix of size
        # (dim x numActions)
        self._degree = degree
        self._numActions = numActions
        self._features = (degree + 1) ** state_size
        self._state_size = state_size
        self._theta = np.zeros((numActions, self._features))

    @property
    def parameters(self) -> np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()

    @parameters.setter
    def parameters(self, p: np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state: np.ndarray, action=None) -> Union[float, np.ndarray]:
        return self.getActionProbabilities(state) if action == None else self.getActionProbabilities(state)[action]

    def samplAction(self, state: np.ndarray) -> int:
        """
        Samples an action to take given the state provided.

        output:
            action -- the sampled action
        """
        prob = self.getActionProbabilities(state)
        action = np.random.choice(self._numActions, p=prob)
        return action

    def fourier(self, state:np.ndarray):
        # cos(pi*C*s)
        C = []
        for i in range(self._features):
            if i == 0:
                Ci = np.zeros(self._state_size)
                C.append(Ci)
                continue
            cnt = i
            t = []
            while cnt != 0:
                remain = cnt % self._degree
                cnt = int(cnt / self._degree)
                t.append(remain)
            if len(t) < self._features:
                Ci = np.zeros(self._features - len(t)).tolist()
            else:
                Ci = []
            Ci.extend(t[::-1])
            C.append(Ci)
        print(np.array(C).shape)
        phi = np.array([np.cos(np.pi * np.dot(state, C[i])) for i in range(self._features)])
        return phi

    def getActionProbabilities(self, state: np.ndarray) -> np.ndarray:
        phi = self.fourier(state)
        sum = np.sum(np.exp(np.dot(self._theta, phi)))
        prob = np.exp(np.dot(self._theta, phi)) / sum
        return prob
