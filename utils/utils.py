import json
import random
import torch

from .model_utils import load_model


def set_seed(seed):
  """Configure random, torch seed for deterministic results"""
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def load_config(path, key=None):
  """
  Load the config file for running experiments. 
  
  If key is provided, return the value of the key in the config. 
  """
  with open(path, "r", encoding="utf-8") as file:
    config = json.load(file)
  if key is None:
    return config
  else:
    return config[key]


def load_text(path):
  with open(path, "r", encoding="utf-8") as f:
    text = f.read()
  return text


def proportion_of_symbols(text):
  """Calculate the proportion of symbols in text"""
  return sum(1 for char in text if not char.isalnum())/len(text)


def prepare_model(model_path):
  """Load a transformer model, put it on cuda if available, and set it to eval model"""
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model, tokenizer, config = load_model(model_path, map_location=device)
  model.eval()

  return model, tokenizer, config