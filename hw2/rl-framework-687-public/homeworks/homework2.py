import numpy as np
from IPython import embed
from rl687.policies.tabular_softmax import TabularSoftmax
from rl687.policies.linear_policy import LinearSoftmax

from rl687.agents.cem import CEM
from rl687.agents.fchc import FCHC
from rl687.agents.ga import GA

from rl687.environments.gridworld import Gridworld
from rl687.environments.cartpole import Cartpole
import matplotlib
from typing import Callable

matplotlib.rcParams['pdf.fonttype'] = 42  # avoid type 3 fonts
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import json
from matplotlib.backends.backend_pdf import PdfPages


def runEnvironment_gridworld(policy, numeps=10000):
    returns = np.zeros(numeps)

    grid = Gridworld()
    for ep in range(numeps):
        grid.reset()
        step = 0
        g = 0
        while not grid.isEnd:
            action = policy.samplAction(grid.state)
            s, r, e = grid.step(action)
            g += (grid.gamma ** step) * r
            step += 1
            if step > 200:
                g = -50
                break
        returns[ep] = g
    return returns

def runEnvironment_carpole(policy, numeps=10000):
    returns = np.zeros(numeps)

    env = Cartpole()
    for ep in range(numeps):
        env.reset()
        step = 0
        g = 0
        while not env.isEnd:
            action = policy.samplAction(env.state)
            s, r, e = env.step(action)
            g += (env.gamma ** step) * r
            step += 1
            if step > 200:
                g = -50
                break
        returns[ep] = g
    return returns

def problem1(para: dict, trails: int = 50):
    """
    Apply the CEM algorithm to the More-Watery 687-Gridworld. Use a tabular
    softmax policy. Search the space of hyperparameters for hyperparameters
    that work well. Report how you searched the hyperparameters,
    what hyperparameters you found worked best, and present a learning curve
    plot using these hyperparameters, as described in class. This plot may be
    over any number of episodes, but should show convergence to a nearly
    optimal policy. The plot should average over at least 500 trials and
    should include standard error or standard deviation error bars. Say which
    error bar variant you used.
    """
    sigma = para['sigma']
    popSize = para['popSize']
    numElite = para['numElite']
    numEpisodes = para['numEpisodes']
    epsilon = para['epsilon']
    mean_return_log = []

    print('sigma:{}\tpopSize:{}\tnumElite:{}\tnumEpisodes:{}\tepsilon:{}'.format(sigma, popSize, numElite, numEpisodes,
                                                                                 epsilon))

    def evaluate(theta, numEpisodes):
        eva_policy = TabularSoftmax(25, 4)
        eva_policy.parameters = theta
        returns = runEnvironment_gridworld(eva_policy, numEpisodes)
        mean_return = np.mean(returns)
        mean_return_log.append(mean_return)
        # print(mean_return)
        return mean_return

    policy = TabularSoftmax(25, 4)
    agent = CEM(theta=policy.parameters, sigma=sigma, popSize=popSize, numElite=numElite, numEpisodes=numEpisodes,
                evaluationFunction=evaluate, epsilon=epsilon)
    for i in range(trails):
        policy.parameters = agent.train()
        print('Episode {} finished'.format(i))

    return mean_return_log


def problem2(para: dict, trails: int = 50):
    """
    Repeat the previous question, but using first-choice hill-climbing on the
    More-Watery 687-Gridworld domain. Report the same quantities.
    """
    sigma = para['sigma']
    numEpisodes = para['numEpisodes']
    mean_return_log = []

    print('sigma:{}\tnumEpisodes:{}\t'.format(sigma, numEpisodes))

    def evaluate(theta, numEpisodes):
        eva_policy = TabularSoftmax(25, 4)
        eva_policy.parameters = theta
        returns = runEnvironment_gridworld(eva_policy, numEpisodes)
        mean_return = np.mean(returns)
        mean_return_log.append(mean_return)
        # print(mean_return)
        return mean_return

    policy = TabularSoftmax(25, 4)
    agent = FCHC(theta=policy.parameters, sigma=sigma, numEpisodes=numEpisodes,
                evaluationFunction=evaluate)
    for i in range(trails):
        policy.parameters = agent.train()
        print('Episode {} finished'.format(i))

    return mean_return_log


def problem3(para: dict, trails: int = 50):
    """
    Repeat the previous question, but using the GA (as described earlier in
    this assignment) on the More-Watery 687-Gridworld domain. Report the same
    quantities.
    """

    popSize = para['popSize']
    numElite = para['numElite']
    numEpisodes = para['numEpisodes']
    alpha = para['alpha']
    print('popSize:{}\tnumElite:{}\tnumEpisodes:{}\talpha:{}'.format(popSize, numElite, numEpisodes,
                                                                                 alpha))
    mean_return_log = []
    def evaluate(theta, numEpisodes):
        eva_policy = TabularSoftmax(25, 4)
        eva_policy.parameters = theta
        returns = runEnvironment_gridworld(eva_policy, numEpisodes)
        mean_return = np.mean(returns, axis=0)
        mean_return_log.append(mean_return)
        # print(mean_return)
        return mean_return

    def initPopulation(popSize:int):
        population = np.random.normal(0, 1, (popSize, 25*4)) # Initialize randomly
        return population

    # policy = TabularSoftmax(25, 4)
    agent = GA(populationSize=popSize, numElite=numElite, numEpisodes=numEpisodes,
                evaluationFunction=evaluate, alpha=alpha, initPopulationFunction=initPopulation)
    for i in range(trails):
        agent.train()
        print('Episode {} finished'.format(i))

    return mean_return_log


def problem4(para: dict, trails: int = 50):
    """
    Repeat the previous question, but using the cross-entropy method on the
    cart-pole domain. Notice that the state is not discrete, and so you cannot
    directly apply a tabular softmax policy. It is up to you to create a
    representation for the policy for this problem. Consider using the softmax
    action selection using linear function approximation as described in the notes.
    Report the same quantities, as well as how you parameterized the policy.

    """
    sigma = para['sigma']
    popSize = para['popSize']
    numElite = para['numElite']
    numEpisodes = para['numEpisodes']
    epsilon = para['epsilon']
    mean_return_log = []

    print('sigma:{}\tpopSize:{}\tnumElite:{}\tnumEpisodes:{}\tepsilon:{}'.format(sigma, popSize, numElite, numEpisodes,
                                                                                 epsilon))

    def evaluate(theta, numEpisodes):
        eva_policy = LinearSoftmax(4, 2, 2)
        eva_policy.parameters = theta
        returns = runEnvironment_carpole(eva_policy, numEpisodes)
        mean_return = np.mean(returns)
        mean_return_log.append(mean_return)
        # print(mean_return)
        return mean_return

    policy = LinearSoftmax(4, 2, 2)
    agent = CEM(theta=policy.parameters, sigma=sigma, popSize=popSize, numElite=numElite, numEpisodes=numEpisodes,
                evaluationFunction=evaluate, epsilon=epsilon)
    for i in range(trails):
        policy.parameters = agent.train()
        print('Episode {} finished'.format(i))

    return mean_return_log


def problem5(para: dict, trails: int = 50):
    """
    Repeat the previous question, but using first-choice hill-climbing (as
    described in class) on the cart-pole domain. Report the same quantities
    and how the policy was parameterized.

    """
    sigma = para['sigma']
    numEpisodes = para['numEpisodes']
    mean_return_log = []

    print('sigma:{}\tnumEpisodes:{}\t'.format(sigma, numEpisodes))

    def evaluate(theta, numEpisodes):
        eva_policy = LinearSoftmax(4, 2, 2)
        eva_policy.parameters = theta
        returns = runEnvironment_carpole(eva_policy, numEpisodes)
        mean_return = np.mean(returns)
        mean_return_log.append(mean_return)
        # print(mean_return)
        return mean_return

    policy = LinearSoftmax(4, 2, 2)
    agent = FCHC(theta=policy.parameters, sigma=sigma, numEpisodes=numEpisodes,
                 evaluationFunction=evaluate)
    for i in range(trails):
        policy.parameters = agent.train()
        print('Episode {} finished'.format(i))

    return mean_return_log


def problem6(para: dict, trails: int = 50):
    """
    Repeat the previous question, but using the GA (as described earlier in
    this homework) on the cart-pole domain. Report the same quantities and how
    the policy was parameterized.
    """
    popSize = para['popSize']
    numElite = para['numElite']
    numEpisodes = para['numEpisodes']
    alpha = para['alpha']
    print('popSize:{}\tnumElite:{}\tnumEpisodes:{}\talpha:{}'.format(popSize, numElite, numEpisodes,
                                                                     alpha))
    mean_return_log = []

    def evaluate(theta, numEpisodes):
        eva_policy = LinearSoftmax(4, 2, 2)
        eva_policy.parameters = theta
        returns = runEnvironment_carpole(eva_policy, numEpisodes)
        mean_return = np.mean(returns, axis=0)
        mean_return_log.append(mean_return)
        # print(mean_return)
        return mean_return

    def initPopulation(popSize: int):
        population = np.random.normal(0, 1, (popSize, 2*81))  # Initialize randomly
        return population

    agent = GA(populationSize=popSize, numElite=numElite, numEpisodes=numEpisodes,
               evaluationFunction=evaluate, alpha=alpha, initPopulationFunction=initPopulation)
    for i in range(trails):
        agent.train()
        print('Episode {} finished'.format(i))

    return mean_return_log


def rand_select_parameters(para: dict, key_list:list):
    para_dict = {}
    for key in key_list:
        choice = np.random.choice(para[key])
        para_dict[key] = choice
    return para_dict


def repeat_experiment(problem: Callable, para: dict, numEpisodes: int = 3, N:int=50):
    return_log = []
    for i in range(numEpisodes):
        mean_return = problem(para, N)
        return_log.append(mean_return)
    return return_log


def search_parameters(problem: Callable, para_list: dict, key_list:list, iterations: int = 10, N: int = 50):
    return_log = []
    for i in range(iterations):
        rand_para = rand_select_parameters(para_list, key_list)
        mean_return = problem(rand_para, N)
        return_log.append([rand_para, mean_return])

    # Calculate the mean for each final 50 episodes' returns
    f = lambda l: (l[0], np.mean(np.array(l[1][-50:])))
    final_mean_return_list = [f(r) for r in return_log]
    final_mean_return_list.sort(key=lambda sample: sample[1], reverse=True)
    best_para = final_mean_return_list[0][0]

    repeated_return_list = repeat_experiment(problem, best_para, N=N)
    mean_returns = np.mean(repeated_return_list, axis=0)
    errs = np.sqrt(np.var(repeated_return_list, axis=0))

    print('best parameter:{}'.format(best_para))
    fig = plt.figure()
    plt.errorbar(np.linspace(0, mean_returns.size, mean_returns.size), mean_returns, yerr=errs, marker='s', mfc='red',
         mec='green', ms=2, mew=1)
    plt.xlabel('Episode Num')
    plt.ylabel('Total return')
    plt.show()
    return best_para, (mean_returns, errs)

def main():
    # Search parameters in Problems
    para_range = {}
    para_range['sigma'] = np.linspace(0.5, 5, 9)
    para_range['popSize'] = np.linspace(10, 50, 5, dtype=np.int32)
    para_range['numElite'] = np.linspace(3, 8, 5, dtype=np.int32)
    para_range['numEpisodes'] = np.linspace(10, 50, 5, dtype=np.int32)
    para_range['epsilon'] = np.linspace(0.5, 2, 5)
    para_range['alpha'] = np.linspace(1, 3, 5)

    p1_keys = ['sigma', 'popSize', 'numElite', 'numEpisodes', 'epsilon']
    best_para, (mean_returns, errs) = search_parameters(problem1, para_range, p1_keys)
    with open('./p1_best_para.json', 'w') as file:
        json.dump(best_para, file)
    with open('./p1_mean_returns.json', 'w') as file:
        json.dump(mean_returns.tolist(), file)
    with open('./p1_errs.json', 'w') as file:
        json.dump(errs.tolist(), file)

    p2_keys = ['sigma','numEpisodes']
    best_para, (mean_returns, errs) = search_parameters(problem2, para_range, key_list=p2_keys, N=500)
    with open('./p2_best_para.json', 'w') as file:
        json.dump(best_para, file)
    with open('./p2_mean_returns.json', 'w') as file:
        json.dump(mean_returns.tolist(), file)
    with open('./p2_errs.json', 'w') as file:
        json.dump(errs.tolist(), file)

    p3_keys = ['sigma', 'popSize', 'numElite', 'numEpisodes', 'alpha']
    best_para, (mean_returns, errs) = search_parameters(problem3, para_range, key_list=p3_keys, N=10)
    with open('./p3_best_para.json', 'w') as file:
        json.dump(best_para, file)
    with open('./p3_mean_returns.json', 'w') as file:
        json.dump(mean_returns.tolist(), file)
    with open('./p3_errs.json', 'w') as file:
        json.dump(errs.tolist(), file)

    p4_keys = ['sigma', 'popSize', 'numElite', 'numEpisodes', 'epsilon']
    best_para, (mean_returns, errs) = search_parameters(problem4, para_range, p4_keys, iterations=10, N=25)
    with open('./p4_best_para.json', 'w') as file:
        json.dump(best_para, file)
    with open('./p4_mean_returns.json', 'w') as file:
        json.dump(mean_returns.tolist(), file)
    with open('./p4_errs.json', 'w') as file:
        json.dump(errs.tolist(), file)

    p5_keys = ['sigma','numEpisodes']
    best_para, (mean_returns, errs) = search_parameters(problem5, para_range, key_list=p5_keys, N=500)
    with open('./p5_best_para.json', 'w') as file:
        json.dump(best_para, file)
    with open('./p5_mean_returns.json', 'w') as file:
        json.dump(mean_returns.tolist(), file)
    with open('./p5_errs.json', 'w') as file:
        json.dump(errs.tolist(), file)

    p6_keys = ['sigma', 'popSize', 'numElite', 'numEpisodes', 'alpha']
    best_para, (mean_returns, errs) = search_parameters(problem6, para_range, key_list=p6_keys, N=10)
    with open('./p6_best_para.json', 'w') as file:
        json.dump(best_para, file)
    with open('./p6_mean_returns.json', 'w') as file:
        json.dump(mean_returns.tolist(), file)
    with open('./p6_errs.json', 'w') as file:
        json.dump(errs.tolist(), file)
if __name__ == "__main__":
    main()
