import copy
import random

from tqdm import tqdm
from utils.population_utils import generate_population


def select(population, fitness_func):
  sorted_pop = sorted(population, key=lambda x: fitness_func(list(x)), reverse=True)
  return sorted_pop


def crossover(parents):
  pos = random.randint(1, len(parents) - 1)  # offsprings have both parents' genes
  offspring = parents[0][:pos] + parents[1][pos:]
  return offspring


def mutate(individual, genes):
  times = random.randint(1, 10) # mutate 1 to 10 times
  mutation_points = random.sample(range(len(individual)), k=times)
  for i in mutation_points:
    new_gene = random.sample(genes, k=1)[0] # uniformly sample genes
    individual[i] = new_gene
  return individual


def genetic_algorithm(n, d, num_of_gen, genes, fitness_func):
  population = generate_population(n, d, genes)
  with tqdm(range(num_of_gen), unit='gen') as pbar:
    for _ in pbar:
      pbar.set_description("Select half of the population with higher fitness")
      population = select(population, fitness_func)
      selected_pop = population[0:len(population) // 2]

      pbar.set_description("Crossovers")
      crossover_times = n - len(selected_pop)
      offsprings = []
      for _ in range(crossover_times):
        parents = random.sample(selected_pop, k=2)
        offspring = crossover(parents)
        assert len(offspring) == d
        offsprings.append(offspring)
      population = selected_pop + offsprings

      pbar.set_description("Mutation")
      for i in range(len(population)):
        if random.random() > 0.5:
          population[i] = mutate(population[i], genes)

  return population
