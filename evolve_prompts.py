import copy
import json
import torch

from pathlib import Path

from constants import MODEL_PATH, CONFIG_PATH
from genetic_algorithm import genetic_algorithm
from utils.inference_utils import generate
from utils.utils import set_seed, proportion_of_nonspace_symbols, prepare_model, load_config, store_output


if __name__ == "__main__":
  model, tokenizer, config = prepare_model(MODEL_PATH)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  output_path = Path("outputs/ga")

  config = load_config(CONFIG_PATH, "ga")
  n = config["n"]
  d = config["d"]
  seed = config["seeds"][0]
  gen = config["gen"]
  set_seed(seed)
  output = copy.deepcopy(config)

  def fitness(individual):
    individual = list(individual)
    output_token_ids = generate(model, individual, length=3*d, device=device)
    output_text = tokenizer.decode(output_token_ids)
    return proportion_of_nonspace_symbols(output_text)

  print("Config: ", json.dumps(config, indent=2))
  genes = list(tokenizer.vocab_to_ind.values())
  population = genetic_algorithm(n, d, gen, genes, fitness)

  resampled_fitness_scores = [fitness(i) for i in population]
  score_tensor = torch.tensor(resampled_fitness_scores)
  i_with_max_f = int(torch.argmax(score_tensor).item())
  best_fit_prompt_tokens = population[i_with_max_f]
  best_fit_prompt_output = tokenizer.decode(generate(model, best_fit_prompt_tokens, 3*d))

  output["resampled-max-f"] = torch.max(score_tensor).item()
  output["resampled-average-f"] = torch.mean(score_tensor).item()
  output["resampled-median-f"] = torch.median(score_tensor).item()
  output["resampled-std-f"] = torch.std(score_tensor).item()
  output["resampled-fitness-scores"] = resampled_fitness_scores
  output["fitness-function"] = proportion_of_nonspace_symbols.__name__
  output["resampled-best-fit-prompt"] = tokenizer.decode(best_fit_prompt_tokens)
  output["resampled-best-fit-prompt-output"] = best_fit_prompt_output

  store_output(output, output_path / f"n{n}-d{d}.json")
