import copy
import json
import random
import torch

from pathlib import Path

from constants import CONFIG_PATH, MODEL_PATH
from fitness import get_population_fitness_func
from utils.inference_utils import generate_tensor
from utils.population_utils import create_population_tensor
from utils.utils import set_seed, prepare_model, proportion_of_nonspace_symbols, load_config, load_text, store_output


if __name__ == "__main__":
  model, tokenizer, config = prepare_model(MODEL_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(device)
  output_path = Path("outputs/baseline")

  config = load_config(CONFIG_PATH, "baseline")
  training_data_path = config["training-data-path"]
  text = load_text(training_data_path)
  text_length = len(text)

  # The baseline of everything - the proportion of symbols in the text
  store_output(
    {"symbol-proportion-in-dataset": proportion_of_nonspace_symbols(text)}, 
    "outputs/symbol-proportion.json"
  )

  n = config["n"]
  d = config["d"]
  D = 4 * d + 1  # total dimension of the population tensor
  seed = config["seeds"][0]
  output = copy.deepcopy(config)
  genes = torch.tensor(list(tokenizer.vocab_to_ind.values()), device=device)
  set_seed(seed)

  print("Config: ", json.dumps(config, indent=2))
  # Sample text tokens
  sampled_text_tokens = []
  for _ in range(n):
    s = random.randint(0, text_length - d)
    sampled_text_tokens.append(tokenizer.encode(text[s: s+d]))
  sampled_text_tokens = torch.tensor(sampled_text_tokens, device=device)

  # Evaluate the text population as the baseline
  population = create_population_tensor(n, D, genes=genes, device=device)
  population[:, :d] = sampled_text_tokens
  population_fitness = get_population_fitness_func(model, tokenizer)
  scored_population = population_fitness(population)

  # Record the results from the baseline
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

  store_output(output, output_path / f"n{n}-d{d}.json")
