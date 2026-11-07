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

  decoding_strategy = make_decoding_strategy(decoding_strategy_name)
  with torch.no_grad():
    for _ in range(length):
      inputs = torch.tensor(token_indx).to(device)
      if len(inputs.size()) == 1:  # input is only a list
        inputs = inputs.unsqueeze(0)

      output = model(inputs, inputs) # (1, T, VOCAB)
      output = output[:, -1, :] # (1, VOCAB)
      output = decoding_strategy(output)

      item = output.item()
      generated_token_index.append(item)
      token_indx.append(item)

  return generated_token_index