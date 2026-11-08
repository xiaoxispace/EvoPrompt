
import argparse
import torch

from pathlib import Path

from genetic_algorithm import genetic_algorithm
from utils.inference_utils import generate
from utils.utils import set_seed, proportion_of_nonspace_symbols, prepare_model


if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-n', type=int, default=66)
  argparser.add_argument('-d', type=int, default=33)
  argparser.add_argument('-g', '--num-of-gen', type=int, default=9)
  parser = argparser.parse_args()
  seed = 496
  set_seed(seed)

  n = parser.n
  d = parser.d
  num_of_gen = parser.num_of_gen

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model_path = Path("checkpoints/ctx384-3e-5/best_model_ctx384_d256_L3_H4_lr3e-05_dropout1e-01_e47.pth")
  model, tokenizer, config = prepare_model(model_path)
  context_size = config['block_size']

  def fitness(individual):
    individual = list(individual)
    output_token_ids = generate(model, individual, length=3*d, device=device)
    output_text = tokenizer.decode(output_token_ids)
    return proportion_of_nonspace_symbols(output_text)

  genes = list(tokenizer.vocab_to_ind.values())
  max_f, individual_with_max_f, population = genetic_algorithm(n, d, num_of_gen, genes, fitness)
  best_output_text = tokenizer.decode(generate(model, individual_with_max_f, 3*d))
  print("Max fitness score: ", max_f)
  print("Individual with max f:\n", tokenizer.decode(individual_with_max_f))
  print("Model output given best individual:\n", best_output_text)
