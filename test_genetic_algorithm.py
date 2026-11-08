import copy
import pytest

from utils.utils import set_seed, proportion_of_symbols
from genetic_algorithm import select, crossover, mutate, genetic_algorithm


@pytest.fixture(scope="module")
def deterministic_result():
  set_seed(496)


def test_selection():
  population = [
    "asfdjlkaelc",
    "asfdjlkael!"
  ]

  new_pop = select(population, proportion_of_symbols)
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

  max_f, individual_with_max_f, population = genetic_algorithm(n, d, num_of_gen, genes, fitness)
  assert len(population) == n
  assert max_f == 30
  assert individual_with_max_f == [3] * d
