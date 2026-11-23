import copy
import json
import torch

from pathlib import Path

from constants import CONFIG_PATH, MODEL_PATH
from fitness import get_population_fitness_func
from utils.population_utils import create_population_tensor
from utils.utils import set_seed, prepare_model, proportion_of_nonspace_symbols, load_config, store_output, nonspace_symbol_set


if __name__ == "__main__":
  model, tokenizer, config = prepare_model(MODEL_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(device)
  output_path = Path("outputs/uniform")

  config = load_config(CONFIG_PATH, "uniform")
  n = config["n"]
  d = config["d"]
  D = 4 * d + 1  # total dimension of the population tensor
  seed = config["seeds"][0]
  set_seed(seed)

  population_fitness_func = get_population_fitness_func(model, tokenizer)
  vocab_to_ind = tokenizer.vocab_to_ind

  print("Config: ", json.dumps(config, indent=2))
  print("Evaluate only the population sampled from nonspace symbols")
  genes = torch.tensor(list(i for k, i in vocab_to_ind.items() if k in nonspace_symbol_set), device=device)
  population = create_population_tensor(n, D, genes=genes, device=device)
  scored_population = population_fitness_func(population)

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
  store_output(output, output_path / f"n{n}-d{d}-symbol.json")

  print("Evaluate the population sampled from the entire vocabulary")
  genes = torch.tensor(list(i for k, i in vocab_to_ind.items()), device=device)
  population = create_population_tensor(n, D, genes=genes, device=device)
  scored_population = population_fitness_func(population)

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
  store_output(output, output_path / f"n{n}-d{d}-vocab.json")
