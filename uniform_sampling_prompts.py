import copy
import torch

from pathlib import Path

from constants import CONFIG_PATH, MODEL_PATH
from utils.inference_utils import generate
from utils.population_utils import generate_population, evaluate_population
from utils.utils import set_seed, prepare_model, proportion_of_nonspace_symbols, load_config, store_output, nonspace_symbol_set


if __name__ == "__main__":
  model, tokenizer, config = prepare_model(MODEL_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  output_path = Path("outputs/uniform")

  config = load_config(CONFIG_PATH, "uniform")
  n = config["n"]
  d = config["d"]
  seed = config["seeds"][0]
  set_seed(seed)

  def fitness_func(individual):
    output_token_ids = generate(model, individual, length = 3*d, device=device)
    output_text = tokenizer.decode(output_token_ids)
    return proportion_of_nonspace_symbols(output_text)

  vocab_to_ind = tokenizer.vocab_to_ind

  print("Evaluate only the population sampled from nonspace symbols")
  genes = list(i for k, i in vocab_to_ind.items() if k in nonspace_symbol_set)
  population = generate_population(n, d, genes)
  max_f, best_fit_prompt_tokens, fitness_scores = evaluate_population(population, fitness_func)
  output = copy.deepcopy(config)
  output["max-f"] = max_f
  output["fitness-function"] = proportion_of_nonspace_symbols.__name__
  output["best-fit-prompt"] = tokenizer.decode(best_fit_prompt_tokens)
  output["best-fit-prompt-output"] = tokenizer.decode(generate(model, best_fit_prompt_tokens, length = 3*d, device=device))
  store_output(output, output_path / f"n{n}-d{d}-symbol.json")

  print("Evaluate the population sampled from the entire vocabulary")
  genes = list(i for _, i in vocab_to_ind.items())
  population = generate_population(n, d, genes)
  max_f, best_fit_prompt_tokens, fitness_scores = evaluate_population(population, fitness_func)
  output = copy.deepcopy(config)
  output["max-f"] = max_f
  output["fitness-function"] = proportion_of_nonspace_symbols.__name__
  output["best-fit-prompt"] = tokenizer.decode(best_fit_prompt_tokens)
  output["best-fit-prompt-output"] = tokenizer.decode(generate(model, best_fit_prompt_tokens, length = 3*d, device=device))
  store_output(output, output_path / f"n{n}-d{d}-vocab.json")
