import random
import numpy as np
import torch
from config.base_config import BaseConfig


from builder import Builder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    baseconfig = BaseConfig()

    # Set global running seed
    set_seed(baseconfig.config["yaml-config"]['env']['seed'])

    # Start assembling RL and training process
    Builder(baseconfig).build().train()


if __name__ == "__main__":
    main()
