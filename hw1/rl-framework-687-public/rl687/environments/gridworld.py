import numpy as np
from .skeleton import Environment
import random
import math

random.seed(38294238.80)

class Gridworld(Environment):
    """
    The Gridworld as described in the lecture notes of the 687 course material. 
    
    Actions: up (0), down (2), left (1), right (3)
    
    Environment Dynamics: With probability 0.8 the robot moves in the specified
        direction. With probability 0.05 it gets confused and veers to the
        right -- it moves +90 degrees from where it attempted to move, e.g., 
        with probability 0.05, moving up will result in the robot moving right.
        With probability 0.05 it gets confused and veers to the left -- moves
        -90 degrees from where it attempted to move, e.g., with probability 
        0.05, moving right will result in the robot moving down. With 
        probability 0.1 the robot temporarily breaks and does not move at all. 
        If the movement defined by these dynamics would cause the agent to 
        exit the grid (e.g., move up when next to the top wall), then the
        agent does not move. The robot starts in the top left corner, and the 
        process ends in the bottom right corner.
        
    Rewards: -10 for entering the state with water
            +10 for entering the goal state
            0 everywhere else
    """

    def __init__(self, startState=0, endState=24, shape=(5,5), obstacles=[12, 17], waterStates=[6, 18, 22]):
        self.initStartState = startState
        self.initEndState = endState
        self.initShape = shape
        self.initObstacles = obstacles
        self.initWaterStates = waterStates

        self.currentState = startState
        self.endState = endState
        self.shape = shape
        self.obstacles = obstacles
        self.waterStates = waterStates
        
        self.timeStep = 0
        self.veerDirection = 0 # 0 for normal, 1 for left, 2 for right
        self.totalReward = 0

        self.upperBounds = [0, 1, 2, 3, 4]
        self.leftBounds = [0, 5, 10, 15, 20]
        self.rightBounds = [4, 9, 14, 19, 24]
        self.lowerBounds = [24, 23, 22, 21, 20]
    @property
    def name(self):
        return 'Gridworld'
        
    @property
    def reward(self):
        if self.currentState in self.waterStates:
            feedback = -10 * math.pow(self.gamma, self.timeStep)
        elif self.currentState == self.endState:
            feedback = 10 * math.pow(self.gamma, self.timeStep)
        else:
            feedback = 0

        self.totalReward += feedback
        return feedback

    @property
    def action(self):
        act = int(4 * random.random())
        return act

    @property
    def isEnd(self):
        if self.currentState == self.endState:
            return True

    @property
    def state(self):
        return self.currentState

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value > 0 and value <= 1:
            self._gamma = value
        else:
            raise ValueError('Gamma should be in 0 to 1')

    def step(self, act):
        rand = random.random()
        if rand <= 0.8:
            pass
        elif rand <= 0.9:
            if random.random() < 0.5:
                # veer left
                self.veerDirection = 1
            else:
                # veer right
                self.veerDirection = 2
        else:
            # stay
            act = None

        # while (not self.isEnd) and self.timeStep < 10000:
        self.timeStep += 1
        old_state = self.currentState
        self.updateState(act)

        feedback = self.reward
        # print('Time {}\nState {} to {}\taction: {}\tVeer: {}\tReward: {}\tTotal reward: {}'.format(self.timeStep, old_state, self.currentState, act, self.veerDirection, feedback,self.totalReward))
        return self.currentState, feedback, self.isEnd

    def reset(self):
        self.__init__(self.initStartState, self.initEndState,
            self.initShape, self.initObstacles, self.initWaterStates)
        
    # def R(self, _state):
    #     """
    #     reward function
        
    #     output:
    #         reward -- the reward resulting in the agent being in a particular state
    #     """

    def updateState(self, act):

        if act is not None:
            if self.veerDirection == 1:
                act += 1
                if act == 4:
                    act = 0
            elif self.veerDirection == 2:
                act -= 1
                if act < 0:
                    act = 3
        # Clear Veer direction
        self.veerDirection = 0

        # Move up
        if act == 0:
            if self.currentState not in self.upperBounds:
                newState = self.currentState - 5
                if newState not in self.obstacles:
                    self.currentState = newState
            else:
                pass
        # Move left
        elif act == 1:
            if self.currentState not in self.leftBounds:
                newState = self.currentState - 1
                if newState not in self.obstacles:
                    self.currentState = newState
            else:
                pass
        # Move down
        elif act == 2:
            if self.currentState not in self.lowerBounds:
                newState = self.currentState + 5
                if newState not in self.obstacles:
                    self.currentState = newState
            else:
                pass
        # Move right
        elif act == 3:
            if self.currentState not in self.rightBounds:
                newState = self.currentState + 1
                if newState not in self.obstacles:
                    self.currentState = newState
            else:
                pass
