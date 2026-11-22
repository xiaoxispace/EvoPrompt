import argparse
import copy
import json
import torch

from pathlib import Path

from constants import MODEL_PATH, CONFIG_PATH
from genetic_algorithm import genetic_algorithm_tensor, genetic_algorithm
from utils.inference_utils import generate, generate_tensor
from utils.utils import set_seed, proportion_of_nonspace_symbols, prepare_model, load_config, store_output
from utils.wandb_utils import setup_wandb


def evolve_prompts(model, tokenizer, config, output_path, n, d, gen, genes):
  def fitness(individual):
    individual = list(individual)
    output_token_ids = generate(model, individual, length=3*d, device='cpu')
    output_text = tokenizer.decode(output_token_ids)
    return proportion_of_nonspace_symbols(output_text)
  population = genetic_algorithm(n, d, gen, genes, fitness)

  resampled_fitness_scores = [fitness(i) for i in population]
  score_tensor = torch.tensor(resampled_fitness_scores)
  i_with_max_f = int(torch.argmax(score_tensor).item())
  best_fit_prompt_tokens = population[i_with_max_f]
  best_fit_prompt_output = tokenizer.decode(generate(model, best_fit_prompt_tokens, 3*d))

  output = copy.deepcopy(config)
  output["resampled-max-f"] = torch.max(score_tensor).item()
  output["resampled-average-f"] = torch.mean(score_tensor).item()
  output["resampled-median-f"] = torch.median(score_tensor).item()
  output["resampled-std-f"] = torch.std(score_tensor).item()
  output["resampled-fitness-scores"] = resampled_fitness_scores
  output["fitness-function"] = proportion_of_nonspace_symbols.__name__
  output["resampled-best-fit-prompt"] = tokenizer.decode(best_fit_prompt_tokens)
  output["resampled-best-fit-prompt-output"] = best_fit_prompt_output

  store_output(output, output_path / f"n{n}-d{d}-g{gen}.json")


def evolve_prompts_tensor(model, tokenizer, config, output_path, n, d, gen, genes, device='cuda'):
  def population_fitness(population, scored_population=None):
    generate_tensor(model, population, start=d, length=3*d)
    if scored_population is None:
      scored_population = population.clone().to(torch.float32)
    for i in range(len(population)):
      output_token_ids = population[i, d:4*d]
      output_text = tokenizer.decode(output_token_ids.tolist())
      scored_population[i, -1] = proportion_of_nonspace_symbols(output_text)
    return scored_population
  scored_population = genetic_algorithm_tensor(n, d, gen, genes, population_fitness, device=device)

  i_with_max_f = int(torch.argmax(scored_population[:, -1]).item())
  best_fit_prompt = tokenizer.decode(scored_population[i_with_max_f, :d].to(torch.int64).tolist())
  best_fit_prompt_output = tokenizer.decode(scored_population[i_with_max_f, d:4*d].to(torch.int64).tolist())

  output = copy.deepcopy(config)
  output["average-f"] = torch.mean(scored_population[:, -1]).item()
  output["median-f"] = torch.median(scored_population[:, -1]).item()
  output["std-f"] = torch.std(scored_population[:, -1]).item()
  output["max-f"] = scored_population[i_with_max_f, -1].item()
  output["fitness-scores"] = scored_population[:, -1].tolist()
  output["fitness-function"] = proportion_of_nonspace_symbols.__name__
  output["best-fit-prompt"] = best_fit_prompt
  output["best-fit-prompt-output"] = best_fit_prompt_output
  output["device"] = device

  store_output(output, output_path / f"n{n}-d{d}-g{gen}.json")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
                  prog='EvoPrompt',
                  description='Evolve prompts using genetic algorithm')
  parser.add_argument('--enable-wandb', action='store_true')
  args = parser.parse_args()
  enable_wandb = args.enable_wandb

  model, tokenizer, config = prepare_model(MODEL_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  output_path = Path("outputs/ga")

  config = load_config(CONFIG_PATH, "ga")
  setup_wandb(config, enable_wandb, "genetic-algorithm")
  n = config["n"]
  d = config["d"]
  seed = config["seeds"][0]
  gen = config["gen"]
  genes = list(tokenizer.vocab_to_ind.values())
  set_seed(seed)

  print("Config: ", json.dumps(config, indent=2))

  if device == 'cpu':
    model.to(device)
    evolve_prompts(model, tokenizer, config, output_path, n, d, gen, genes, device=device)
  else:
    model.to(device)
    evolve_prompts_tensor(model, tokenizer, config, output_path, n, d, gen, genes, device=device)
