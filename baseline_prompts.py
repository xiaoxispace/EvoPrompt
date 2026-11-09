import copy
import json
import random
import torch

from pathlib import Path

from constants import CONFIG_PATH, MODEL_PATH
from utils.inference_utils import generate
from utils.population_utils import evaluate_population
from utils.utils import set_seed, prepare_model, proportion_of_nonspace_symbols, load_config, load_text, store_output


if __name__ == "__main__":
  model, tokenizer, config = prepare_model(MODEL_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
  seed = config["seeds"][0]
  output = copy.deepcopy(config)
  set_seed(seed)

  def fitness_func(individual):
    output_token_ids = generate(model, individual, length = 3*d, device=device)
    output_text = tokenizer.decode(output_token_ids)
    return proportion_of_nonspace_symbols(output_text)

  print("Config: ", json.dumps(config, indent=2))
  max_f = 0
  best_prompt_tokens = []
  population = []
  for _ in range(n):
    s = random.randint(0, text_length - d)
    population.append(tokenizer.encode(text[s: s+d]))
  f, prompt_tokens, fitness_scores = evaluate_population(population, fitness_func)

  if f > max_f:
    max_f = f
    best_prompt_tokens = prompt_tokens

  print("  Max fitness score: ", max_f)
  print("  Output text with max_f: ", tokenizer.decode(best_prompt_tokens))

  output["max-f"] = max_f
  output["fitness-function"] = proportion_of_nonspace_symbols.__name__
  output["best-fit-prompt"] = tokenizer.decode(best_prompt_tokens)
  output["best-fit-prompt-output"] = tokenizer.decode(generate(model, best_prompt_tokens, length = 3*d, device=device))

  store_output(output, output_path / f"n{n}-d{d}-s{seed}.json")
