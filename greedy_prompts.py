import copy
import json
import torch

from pathlib import Path

from constants import CONFIG_PATH, MODEL_PATH
from utils.inference_utils import generate
from utils.population_utils import generate_population, evaluate_population
from utils.utils import set_seed, prepare_model, proportion_of_nonspace_symbols, load_config, store_output, nonspace_symbol_set

map_symbol_to_name = {
    '\u2019': 'right_single_quote',
    '[': 'left_square_bracket',
    ']': 'right_square_bracket',
    '?': 'question_mark',
    '>': 'greater_than',
    '<': 'less_than',
    ':': 'colon',
    ';': 'semicolon',
    '/': 'forward_slash',
    '.': 'period',
    '-': 'hyphen',
    ',': 'comma',
    ')': 'right_parenthesis',
    '(': 'left_parenthesis',
    '\'': 'apostrophe',
    '&': 'ampersand',
    '"': 'double_quote',
    '!': 'exclamation_mark',
}

if __name__ == "__main__":
  model, tokenizer, config = prepare_model(MODEL_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  output_path = Path("outputs/greedy")

  config = load_config(CONFIG_PATH, "greedy")
  n = config["n"]
  d = config["d"]
  seed = config["seeds"][0]
  set_seed(seed)

  def fitness_func(individual):
    output_token_ids = generate(model, individual, length = 3*d, device=device)
    output_text = tokenizer.decode(output_token_ids)
    return proportion_of_nonspace_symbols(output_text)
  vocab_to_ind = tokenizer.vocab_to_ind

  print("Config: ", json.dumps(config, indent=2))
  print("Evaluate only the population by greedily using one symbol as the prompts")

  for symbol, name in map_symbol_to_name.items():
    genes = [vocab_to_ind[symbol]]
    population = generate_population(n, d, genes)
    max_f, best_fit_prompt_tokens, fitness_scores = evaluate_population(population, fitness_func)
    output = copy.deepcopy(config)
    score_tensor = torch.tensor(fitness_scores)
    output["max-f"] = torch.max(score_tensor).item()
    output["average-f"] = torch.mean(score_tensor).item()
    output["median-f"] = torch.median(score_tensor).item()
    output["std-f"] = torch.std(score_tensor).item()
    output["symbol"] = symbol
    output["fitness-scores"] = fitness_scores
    output["fitness-function"] = proportion_of_nonspace_symbols.__name__
    output["resampled-best-fit-prompt-output"] = tokenizer.decode(generate(model, best_fit_prompt_tokens, length = 3*d, device=device))
    store_output(output, output_path / f"n{n}-d{d}-symbol-{name}.json")