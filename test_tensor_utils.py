
import torch

from constants import MODEL_PATH
from utils.inference_utils import generate_tensor
from utils.utils import prepare_model
from utils.population_utils import create_population_tensor


def test_create_population_tensor():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n = 6
  d = 8
  D = d*4 + 1
  _, tokenizer, _ = prepare_model(MODEL_PATH)
  genes = torch.arange(tokenizer.vocab_size, device=device)
  population = create_population_tensor(n, D, genes, device)
  assert population.shape == (n, D)
  assert torch.all(population >= 0) and torch.all(population < tokenizer.vocab_size)


def test_generate_tensor():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n = 6
  d = 8
  D = d*4 + 1
  model, tokenizer, _ = prepare_model(MODEL_PATH)
  population = torch.randint(0, tokenizer.vocab_size, (n, D), device=device)
  original = population.clone()

  model.to(device)
  generate_tensor(model, population, start=d, length=3*d)
  assert torch.all(population[:, :d] == original[:, :d])  # the prompt part remains unchanged
  assert torch.all(population[:, 4*d:] == original[:, 4*d:])  # the score part remains unchanged