import torch


from utils.inference_utils import generate_tensor
from utils.utils import proportion_of_nonspace_symbols


def get_population_fitness_func(model, tokenizer):
  def population_fitness(population, scored_population=None):
    n, D = population.shape
    d = (D-1) // 4  # the prompt size
    generate_tensor(model, population, start=d, length=3*d)  # population[:, d:4*d] are modified in-place
    # create a separate tensor for storing fitness scores in float tensor
    if scored_population is None:
        scored_population = population.clone().to(torch.float32)

    for i in range(n):
        output_token_ids = population[i, d:4*d]
        output_text = tokenizer.decode(output_token_ids.tolist())
        scored_population[i, -1] = proportion_of_nonspace_symbols(output_text)
    return scored_population
  return population_fitness
