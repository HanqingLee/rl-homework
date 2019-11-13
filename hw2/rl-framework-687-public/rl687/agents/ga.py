import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, 
                 initPopulationFunction:Callable, numElite:int=1, numEpisodes:int=10, alpha:float=2.5):
        self._name = "Genetic_Algorithm"
        self._populationSize = populationSize
        self._evaluationFunction = evaluationFunction
        self._initPopulationFunction = initPopulationFunction
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._alpha = alpha

        self._population:np.ndarray = self._initPopulationFunction(self._populationSize)
        self._bestIndividual = None

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._bestIndividual[0]

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        epsilon = np.random.normal(0, 1, size=parent.size)
        child = parent + self._alpha * epsilon
        return child

    def _get_children(self, parents:np.ndarray)->np.ndarray:
        num_children = self._populationSize - parents.shape[0]
        children_list = []
        for cnt in range(num_children):
            selected_parent = parents[np.random.choice(parents.shape[0])]
            child = self._mutate(selected_parent)
            children_list.append(child)
        return np.array(children_list)


    def train(self)->np.ndarray:
        sample_list = []
        for k in range(self._populationSize):
            theta_k = self._population[k]
            object_k = self._evaluationFunction(theta_k, self._numEpisodes)
            sample_list.append([theta_k, object_k])
        sample_list.sort(key=lambda sample: sample[1], reverse=True)
        bestIndividual = sample_list[0]
        if self._bestIndividual is None:
            self._bestIndividual = bestIndividual
        elif bestIndividual[1] > self._bestIndividual[1]:
            self._bestIndividual = bestIndividual
        elite_sample = sample_list[:self._numElite]
        parents = np.array([sample[0] for sample in elite_sample])
        children = self._get_children(parents)
        next_gen = np.vstack((parents, children))
        self._population = next_gen

        return next_gen


    def reset(self)->None:
        self._population = self._initPopulationFunction(self._populationSize)