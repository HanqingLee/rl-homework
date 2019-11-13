import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "Cartpole"
        self.force = 10
        self.absx_range = 3.

        # TODO: properly define the variables below
        self._action = None
        self._reward = 0
        self._isEnd = False
        self._gamma = 1

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # time step
        self._t = 0.0  # total time elapsed  NOTE: you must use this variable

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        self._state = np.array([self._x, self._v, self._theta, self._dtheta])
        return self._state

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        x, v, theta, dtheta = state

        force = (action * 2 - 1) * self.force  # specify the force direction

        ddtheta = (self._g * np.sin(theta) + np.cos(theta) * (
                (-force - self._mp * self._l * np.square(dtheta) * np.sin(theta)) / (self._mc + self._mp))) / (
                          self._l * (4. / 3. - self._mp * np.square(np.cos(theta)) / (self._mc + self._mp)))
        dv = (force + self._mp * self._l * (np.square(dtheta) * np.sin(theta) - ddtheta * np.cos(theta))) / (
                self._mc + self._mp)
        next_state = state + self._dt * np.array([v, dv, dtheta, ddtheta])
        return next_state

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        return 1.

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """

        next_state = self.nextState(self.state, action)
        self._reward = self.R(self.state, action, next_state)
        # Update
        self._x, self._v, self._theta, self._dtheta = next_state
        self._t += self._dt
        self._isEnd = self.terminal()
        self._action = action
        return next_state, self._reward, self.isEnd

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole
        self._action = None
        self._isEnd = False
        self._t = 0.

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        x, _, theta, _ = self.state
        if abs(x) >= self.absx_range:
            self._isEnd = True
        if abs(theta) > np.pi / 12:
            self._isEnd = True
        if self._t > 20:
            self._isEnd = True
        return self.isEnd
