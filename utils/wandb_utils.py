import copy
import wandb


def setup_wandb(config, enable_wandb, experiment_name):
  wandb_config = copy.deepcopy(config)
  wandb_config["experiment"] = experiment_name
  if enable_wandb:
    wandb.init(
      project="evo-prompt",
      config=wandb_config
    )
    run_name = wandb.run.name
  else:
    print("Disable wandb")
    wandb.init(mode="disabled")
    run_name = None

  return run_name