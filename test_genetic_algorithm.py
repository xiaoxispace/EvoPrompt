import copy
import pytest
import torch

from genetic_algorithm import (
  select, crossover, mutate, genetic_algorithm,
  select_tensor, crossover_tensor, mutate_tensor, genetic_algorithm_tensor
)
from utils.utils import set_seed, proportion_of_nonspace_symbols
from utils.wandb_utils import setup_wandb


@pytest.fixture(scope="module")
def deterministic_result():
  set_seed(496)


def test_selection():
  population = [
    "asfdjlkaelc",
    "asfdjlkael!"
  ]

  new_pop = select(population, proportion_of_nonspace_symbols)
  assert new_pop[0] == population[1]
  assert new_pop[1] == population[0]
  assert id(new_pop[0]) == id(population[1])
  assert id(new_pop[1]) == id(population[0])


def test_crossover(deterministic_result):
  parents = [
    "asdjlkjakle",
    "jioejoijosj"
  ]

  offspring = crossover(parents)
  assert offspring != parents[0] and offspring != parents[1], \
    "Offspring should not be the same as parents"
  assert id(offspring) != id(parents[0]) and id(offspring) != id(parents[1]), \
    "Offspring should be a new list"
  assert len(offspring) == len(parents[0]) and len(offspring) == len(parents[1]), \
    f"Offspring should have the same dim as parents, actual: {len(offspring)}"


def test_mutate(deterministic_result):
  population = [["e"] * 10, ["d"] * 10]
  for p in population:
    mp = copy.deepcopy(p)
    mutate(mp, genes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    assert mp != p, "Expect inplace mutation of an individual"


def test_genetic_algorithm_basic(deterministic_result):
  n = 60
  d = 10
  num_of_gen = 90
  genes = [0, 1, 2, 3]
  fitness = sum

  population = genetic_algorithm(n, d, num_of_gen, genes, fitness)
  assert len(population) == n


def test_selection_tensor():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  scored_population = torch.tensor([
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  ], device=device)
  original = scored_population.clone()

  selected_pop = select_tensor(scored_population)
  assert torch.all(selected_pop[0] == original[1])
  assert torch.all(selected_pop[1] == original[0])
  assert id(selected_pop) != id(original)
  assert id(selected_pop) != id(scored_population)  # out-of-place select


def test_crossover_tensor():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  parents_pool = torch.tensor([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ], device=device)
  scored_population = torch.zeros((6, 10), device=device)
  crossover_tensor(parents_pool, scored_population)
  assert torch.all(scored_population[0:3, :] == 0)
  for i in range(3, 6):
    for j in range(10):
      assert scored_population[i, j] == parents_pool[0, j] or scored_population[i, j] == parents_pool[1, j] or scored_population[i, j] == parents_pool[2, j]


def test_mutate_tensor(deterministic_result):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n = 6
  d = 8
  D = d*4 + 1
  scored_population = torch.randint(0, 10,(n, D), device=device)
  original = scored_population.clone()

  genes = torch.tensor([-1, -2, -3, -4, -5, -6, -7, -8, -9], device=device)  # mutate only with negative numbers for identifying mutations
  mutate_tensor(scored_population, genes=genes)
  mutate_mask = (scored_population < 0).any(dim=1).to(device)
  mutated_individuals = scored_population[mutate_mask]
  target_mutations = torch.sum((mutated_individuals < 0)[0]).item()  # the num of mutations of the first mutated individual

  assert len(mutated_individuals) == 4  # based on deterministic results: exactly 4 individuals should be mutated
  assert torch.all((mutated_individuals<0).count_nonzero(dim=1) == target_mutations)  # each mutated individual should have the exact same number of mutations
  assert torch.all(scored_population[:, d:] == original[:, d:])  # the outputs and scores are not mutated


def test_genetic_algorithm_tensor_basic(deterministic_result):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n = 9
  d = 12
  D = d*4 + 1
  num_of_gen = 30
  genes = torch.arange(10, 21, device=device)  # genes are 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
  setup_wandb({}, False, "None")

  # basically, maximizing the sum of the first d genes for each individual
  # however, the fitness score is defined by sampling the d genes uniformly 3*d times and summing the results
  def population_fitness_func(population, scored_population=None):
    n, D = population.shape
    d = (D - 1) // 4
    scored_population = population.clone().to(torch.float32) if scored_population is None else scored_population
    # pytorch uniformly sample from population[i, 0:12] to populate [i, 12: 48]
    indices = torch.randint(0, d, (n, 3*d), device=population.device)
    scored_population[:, d:4*d] = scored_population[:, 0:d].gather(1, indices)
    scored_population[:, 4*d] = torch.sum(scored_population[:, d:4*d], dim=1)

    return scored_population

  scored_population = genetic_algorithm_tensor(n, d, num_of_gen, genes, population_fitness_func, device=device)
  assert scored_population.shape == (n, D)
  assert torch.all(scored_population[:, 4*d] == torch.sum(scored_population[:, d:4*d], dim=1))
  assert torch.all(torch.isin(scored_population[:, 0:4*d], genes))
