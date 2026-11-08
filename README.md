# Evolutionary Prompt Optimization for Symbolic Generation
Evolving character-level prompts to make a 4M Shakespeare Transformer output symbols instead of language.

This project utilizes the models trained from the [Transformer Playground](https://github.com/xiaoxispace/transformer-playground) project. The checkpoint to use is the model that achieves the minimum validation loss, specifically the model with 384 context size and trained with `3e-5` learning rate.


# Experiments

## Baseline

The first baseline to compare is the proportion of nonspace symbols in the dataset. The proportion is approximately 4.5%. The result is calculated by executing `python baseline_prompt.py` and is stored in `outputs/symbol-proportion.json`. 

Another baseline is how many nonspace symbols does the model generate given the text input in the dataset. `baseline_prompt.py` also calculates this by sampling the dataset 100 times and return the prompt that leads to the most nonspace symbol output. The experiment result is stored in `outputs/baseline/n100-d384-s496.json`. 

## Uniform Sampling

An improved approach than sampling the dataset is to sample the vocabulary. The sampling can be done in the entire vocabulary space or only the nonspace vocabulary space. The later tends to assume more symbols in the prompt will cause the model to generate more symbols in the output. 

# Quick Start

## Install Python Environment

1. Install pyenv from [the pyenv public repo](https://github.com/pyenv/pyenv#a-getting-pyenv). Specifically, for linux, 
    1. Update and install system dependencies
        ```bash
        sudo apt update
        sudo apt install -y make build-essential libssl-dev zlib1g-dev \
          libbz2-dev libreadline-dev libsqlite3-dev curl git \
          libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
          libffi-dev liblzma-dev
        ```
    2. Run `curl -fsSL https://pyenv.run | bash` to install pyenv
    3. Setup pyenv for shell with command
        ```bash
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        ```

        Note: If this is a Mac, replace `.bashrc` with `.zshrc` in all three commands above.
    4. Restart shell with command `exec "$SHELL"`
2. Install Python and the required dependencies
    1. Run `pyenv install 3.12.15`
    2. Create the virtual environment `pyenv virtualenv 3.12.15 evo-prompt`
    3. Install the required dependencies `python -m pip install -r requirements.txt`
    4. Note: In an environment powered by CPU such as Mac without MPS or Linux without CUDA, please use the requirements file `requirements-cpu.txt`, which installs native `torch` and `torchvision` pacakges instead of their CUDA counterparts.