"""This file contains the utilities for output evaluations and generations."""
import torch
from tqdm import tqdm


SUPPORTED_DECODING_STRATEGIES = (
  "default", "multinomial", "argmax"
)


def multinomial_decoding(logits):
  output = torch.softmax(logits, dim=-1) # [1, VOCAB]
  output = torch.multinomial(output, num_samples=1)
  return output


def argmax_decoding(logits):
  return torch.argmax(logits)


_NAME_TO_DECODE_STRATEGY = {
  "default": multinomial_decoding,
  "multinomial": multinomial_decoding,
  "argmax": argmax_decoding
}


def make_decoding_strategy(name):
  if name.lower() not in SUPPORTED_DECODING_STRATEGIES:
    raise ValueError(f"Decoding strategy {name} not supported")

  return _NAME_TO_DECODE_STRATEGY[name]


def generate(model, token_indx, length=1000, device='cpu', decoding_strategy_name='default'):
  """Generate the list of output token indices given a list or a batch of list of token indices. """
  generated_token_index = []

  new_inputs = torch.zeros(len(token_indx) + length + 1, dtype=torch.int64).to(device) 
    # +1 for extra buffer in the for-loop
    # without it, the last line in the loop will break at the last iteration
  new_inputs[0: len(token_indx)] = torch.tensor(token_indx, dtype=torch.int64).to(device) # assume token_indx is 1-D
  new_inputs = new_inputs.unsqueeze(0) # form a batch

  decoding_strategy = make_decoding_strategy(decoding_strategy_name)
  with torch.no_grad():
    for s in range(len(token_indx), len(token_indx) + length):
      output = model(new_inputs[:, 0: s], new_inputs[:, 0: s]) # (1, T, VOCAB)
      output = output[:, -1, :] # (1, VOCAB)
      output = decoding_strategy(output)

      item = output.item()
      generated_token_index.append(item)
      new_inputs[:, s+1] = item

  return generated_token_index