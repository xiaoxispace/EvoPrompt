import random
import torch
import wandb

from tqdm import tqdm
from utils.population_utils import generate_population, create_population_tensor
from utils.inference_utils import generate_tensor


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


def select_tensor(population):
  sorted_pop = population[population[:, -1].argsort(descending=True)]
  return sorted_pop


def crossover_tensor(parents_pool, scored_population):
  n, D = scored_population.shape
  p = len(parents_pool)
  k = n - p
  d = (D-1) // 4  # the prompt size

  if k == 0:
    return scored_population
  if p < 2 or d < 2:
    raise ValueError("Requires at least 2 parents and 2 columns")

  device = parents_pool.device

  # Random parent pairs (distinct within each pair)
  idx1 = torch.randint(p, (k,), device=device)
  idx2 = torch.randint(p - 1, (k,), device=device)
  idx2 += (idx2 >= idx1).long()

  # Random crossover points
  points = torch.randint(1, d, (k, 1), device=device)  # (k, 1)

  # Crossover mask
  mask = torch.arange(d, device=device)[None, :] < points  # (k, d)

  # Generate all offspring
  offspring = torch.where(
    mask,
    parents_pool[idx1, 0:d],
    parents_pool[idx2, 0:d]
  )

  scored_population[p:, 0:d] = offspring


def mutate_tensor(scored_population, genes):
    n, D = scored_population.shape
    d = (D-1) // 4  # the prompt size
    device = scored_population.device

    # Randomly select individuals (50% probability)
    rows = torch.where(torch.rand(n, device=device) > 0.5)[0]

    # Same number of mutations for all selected individuals
    times = torch.randint(1, min(10, d) + 1, (1,)).item()

    # Sample distinct mutation positions for each individual
    cols = torch.rand(len(rows), d, device=device).topk(
      times, dim=1
    ).indices

    # Sample replacement genes uniformly
    gene_idx = torch.randint(
      len(genes), (len(rows), times), device=device
    )

    # Mutate in-place
    scored_population[rows[:, None], cols] = genes[gene_idx].to(scored_population.dtype)

    return scored_population


def genetic_algorithm_tensor(n, d, num_of_gen, genes, population_fitness_func, device='cuda'):
  genes = torch.as_tensor(  # genes is just a set (or list) of token indices in vocab; convert to tensor for vectorized operations
    genes,
    dtype=torch.int64,
    device=device
  )
  D = d*4+1
  population = create_population_tensor(n, D, genes, device=device)  # extra dimensions as placeholders for outputs and fitness scores
  scored_population = population_fitness_func(population)
  wandb.log({
    "average-f": torch.mean(scored_population[:, -1]).item(),
    "median-f": torch.median(scored_population[:, -1]).item(),
    "std-f": torch.std(scored_population[:, -1]).item(),
    "max-f": torch.max(scored_population[:, -1]).item(),
  })

  with tqdm(range(1, num_of_gen+1), unit='gen') as pbar:
    for e in pbar:
      pbar.set_description("Select half of the population with higher fitness")
      scored_population = select_tensor(scored_population)
      selected_pop = scored_population[0:len(population) // 2, :]
        # select the top half of the population as parents for crossover
        # new offsprings are assigned directly to the bottom half of scored_population

      pbar.set_description("Crossovers")
      crossover_tensor(selected_pop, scored_population)

      pbar.set_description("Mutation")
      mutate_tensor(scored_population, genes)

      # iterate to the next generation
      population = scored_population.to(torch.int64)
      scored_population = population_fitness_func(population, scored_population)  # operate on the existing scored_population to update fitness scores

      # log fitness statistics to wandb
      wandb.log({
        "average-f": torch.mean(scored_population[:, -1]).item(),
        "median-f": torch.median(scored_population[:, -1]).item(),
        "std-f": torch.std(scored_population[:, -1]).item(),
        "max-f": torch.max(scored_population[:, -1]).item(),
      })
  return scored_population