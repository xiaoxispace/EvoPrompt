import copy
import random

from tqdm import tqdm


def create_individual(d, genes):
  """Uniformly sampling genes to produce an individual"""
  return random.choices(genes, k=d)


def generate_population(n, d, genes):
  """Generate a population of size n with each individual of dimension of d, sampled from genes"""
  return [create_individual(d, genes) for _ in range(n)]


def evaluate_population(population, fitness_func):
  """Return the fintess scores of the population and population's best fit indiviual"""
  fitness_scores = []
  max_f = 0
  best_individual = []
  for individual in tqdm(population, desc='Evaluating population', unit='eval'):
    f = fitness_func(individual)
    fitness_scores.append(f)

    if f > max_f:
      best_individual = copy.deepcopy(individual)
      max_f = f

  return max_f, best_individual, fitness_scores
