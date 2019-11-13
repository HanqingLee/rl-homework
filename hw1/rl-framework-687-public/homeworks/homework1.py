import numpy as np
from rl687.environments.gridworld import Gridworld
import json
import matplotlib.pyplot as plt
import math
import random
random.seed(38294238.80)

def problemA():
    """
    Have the agent uniformly randomly select actions. Run 10,000 episodes.
    Report the mean, standard deviation, maximum, and minimum of the observed 
    discounted returns.
    """
    time_list = []
    reward_list = []
    env = Gridworld()
    env.gamma = 0.9

    episode = 0
    while episode <= 10000:
        episode += 1
        print('Episode {}'.format(episode))
        step = 0
        totalReward = 0
        reached = False
        while True:
            step += 1
            act = env.action
            state, reward, isEnd = Gridworld.step(env, act)
            # reward_list.append(reward)
            totalReward += reward
            if isEnd:
                reached = True
                print('Steps take: {}\tTotal reward: {:.4f}'.format(step, totalReward))
                break
        if not reached:
            episode -= 1
            continue
        Gridworld.reset(env)
        reward_list.append(totalReward)
    print('finished')

    reward_array = np.array(reward_list)
    mean = reward_array.mean()
    std = reward_array.std()
    max = reward_array.max()
    min = reward_array.min()

    print('Mean: {:.2f}\tSTD: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(mean, std, max, min))
    print('Num of reward: {}'.format(len(reward_list)))
    with open('./resultA.json', 'w') as file:
        json.dump(reward_list, file)


def problemB():
    """
    Run the optimal policy that you found for 10,000 episodes. Repor the 
    mean, standard deviation, maximum, and minimum of the observed 
    discounted returns
    """
    # if on the upper edge, move right; if on right edge, move down;
    # else, move right or down
    env = Gridworld()
    env.gamma = 0.9
    episode = 0
    reward_list = []
    # obstacles = [12, 17]
    # waterStates = [6, 18, 22]
    # upperBounds = [0, 1, 2, 3, 4]
    # rightBounds = [4, 9, 14, 19, 24]

    while episode < 10000:
        episode += 1
        print('Episode {}'.format(episode))
        step = 0
        totalReward = 0
        reached = False
        while step < 10000:
            step += 1
            if env.currentState in env.rightBounds:
                act = 2 # Move down
            elif env.currentState in env.upperBounds:
                act = 3 # Move right
            else:
                if random.random() < 0.5:
                    act = 3
                else:
                    act = 2
            # secure = False
            # while not secure:
            #     if act == 2:
            #         nextState = env.currentState + 5
            #         if nextState in env.waterStates or nextState in env.obstacles:
            #             act = 3
            #         else:
            #             secure = True
            #     else:
            #         nextState = env.currentState + 1
            #         if nextState in env.waterStates or nextState in env.obstacles:
            #             act = 2
            #         else:
            #             secure = True

            state, reward, isEnd = Gridworld.step(env, act)
            totalReward += reward
            if isEnd:
                reached = True
                print('Steps take: {}\tTotal reward: {:.4f}'.format(step, totalReward))
                break
        if not reached:
            episode -= 1
            continue
        Gridworld.reset(env)
        reward_list.append(totalReward)
    print('finished')

    reward_array = np.array(reward_list)
    mean = reward_array.mean()
    std = reward_array.std()
    max = reward_array.max()
    min = reward_array.min()

    print('Mean: {:.2f}\tSTD: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(mean, std, max, min))
    print('Num of reward: {}'.format(len(reward_list)))
    with open('./resultB.json', 'w') as file:
        json.dump(reward_list, file)


def problemE():
    env = Gridworld(startState=18)
    env.gamma = 0.9
    episode = 0
    hit = 0
    total_try = 100000
    while episode < total_try:
        episode += 1
        env.timeStep = 8
        while env.timeStep < 19:
            act = env.action
            state, reward, isEnd = Gridworld.step(env, act)
            if isEnd:
                break
        if env.currentState == 21:
            hit += 1
    print('P is {}'.format(hit/total_try))


def CDF(reward_list):
    reward_array = np.array(reward_list)
    lower = math.floor(reward_array.min())
    upper = math.ceil(reward_array.max())

    X = np.linspace(lower, upper, 1000)
    cdf = []
    for [i, x] in enumerate(X):
        if len(cdf) > 0:
            cnt = cdf[-1]
        else:
            cnt = 0
        for [index, reward] in enumerate(reward_list):
            if reward < x:
                cnt += 1
                reward_list.pop(index)
        cdf.append(cnt)
    cdf = np.array(cdf)
    cdf = cdf / reward_array.size

    plt.plot(X, cdf)
    plt.title('CDF of optimal policy')
    plt.xlabel('Reward')
    plt.ylabel('CDF value')
    plt.show()

def quantile(reward_list):
    # reward_array = np.array(reward_list)
    # np.sort(reward_array)
    reward_list.sort()
    n = len(reward_list)
    Q = []
    X = np.linspace(0, 1, 1000)
    X = np.delete(X, -1)
    for x in X:
        index = math.floor((n+1)*x)
        try:
            value = reward_list[index]
            Q.append(value)
        except:
            print('failure for index {}'.format(index))
            break

    plt.plot(X, Q)
    plt.title('Quantile of random policy')
    plt.xlabel('Probability')
    plt.ylabel('Reward')
    plt.show()

def main():
    # problemA()
    # problemB()
    # problemE()
    # reward_list = json.load(open('./resultA.json', 'r'))
    reward_list = json.load(open('./resultB.json', 'r'))

    CDF(reward_list)
    # quantile(reward_list)
main()

        
