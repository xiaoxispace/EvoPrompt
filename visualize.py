import re
import numpy as np

from pathlib import Path
from utils.utils import load_config

import matplotlib.pyplot as plt


pattern = re.compile(r"^n(?P<n>\d+)-d(?P<d>\d+)(?:-(?P<variant>\w+))?\.json$")

def handle_baseline(path):
  files = sorted([f for f in path.iterdir() if f.is_file()], key=lambda f: f.name)
  title = path.name
  data = {}

  for f in files:
    match = pattern.match(f.name)
    if not match:
      raise ValueError("Match not found!!!")
    n = int(match.group("n"))
    d = int(match.group("d"))

    res = load_config(f)
    data[d] = res["max-f"]

  return data, title


def handle_ga(path):
  files = sorted([f for f in path.iterdir() if f.is_file()], key=lambda f: f.name)
  title = path.name
  data = {}

  for f in files:
    match = pattern.match(f.name)
    if not match:
      raise ValueError("Match not found!!!")
    n = int(match.group("n"))
    d = int(match.group("d"))

    res = load_config(f)
    data[d] = res["max-f"]
  return data, title


def handle_greedy(path):
  files = [f for f in path.iterdir() if f.is_file()]
  pass


def handle_uniform(path):
  files = sorted([f for f in path.iterdir() if f.is_file()], key=lambda f: f.name)
  title = path.name
  data = {}

  for f in files:
    match = pattern.match(f.name)
    if not match:
      raise ValueError("Match not found!!!")
    n = int(match.group("n"))
    d = int(match.group("d"))
    variant = match.group("variant") or "default"

    res = load_config(f)
    if d not in data:
      data[d] = res["max-f"]
    else:
      data[d] = max(data[d], res["max-f"])
  return data, title


def handle_symbol_proportion(path):
  res = load_config(path)
  return res["symbol-proportion-in-dataset"]


RESULT_PATHS = {
  Path("outputs/baseline"): handle_baseline,
  Path("outputs/ga"): handle_ga,
  # Path("outputs/greedy"): handle_greedy,
  Path("outputs/uniform"): handle_uniform,
}


color_map = {
  'ga': "#FF6B00",
  'baseline': "#708090",
  'sanity check': "#000000",
  'uniform': "#B0C4DE"
}

if __name__ == "__main__":
  tick_labels = []
  data = []

  for p, f in RESULT_PATHS.items():
    dt, label = f(p)
    data.append(dt)
    tick_labels.append(label)
    dims = sorted(dt.keys())
    x = [str(d) for d in dims]
    y = np.array([ dt[d] for d in dt.keys() ])
    plt.plot(x, y, label=label, color=color_map[label])

  str_x = [str(d) for d in [64, 128, 256, 384]]
  data = handle_symbol_proportion(Path("outputs/symbol-proportion.json"))
  plt.plot(str_x, [data]*4, label='sanity check', color=color_map['sanity check'])

  # plt.xticks([64, 128, 256, 384])
  plt.xlabel("prompt length")
  plt.ylabel("max fintess score")
  plt.title("Performance Comparison")

  plt.legend()
  plt.show()

  # box_plot(data, tick_labels, title="Performance Comparison", xlabel="Benchmark Type", ylabel="Fintess Score")
  pass