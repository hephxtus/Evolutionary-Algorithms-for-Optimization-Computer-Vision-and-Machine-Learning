import numpy as np
import os
import random
import torch
from builder import Builder
from config.eval_config import EvalConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    which_log = 'logs/CartPole-v1'
    save_model_freq = 20
    generation_num = 5000

    log_folders = [f.path for f in os.scandir(which_log) if f.is_dir()]
    for log_path in log_folders:
        # testing file exits, delete it
        print(log_path)
        testing_record = f'{log_path}/test_performance/testing_record.csv'
        if os.path.exists(testing_record):
            os.remove(testing_record)

        for fr in np.arange(save_model_freq, generation_num + save_model_freq, 20, dtype=int):
            # log file not exits, break the loop
            model = f'{log_path}/saved_models/ep_{fr}.pt'
            if not os.path.exists(model):
                break
            eval_config = EvalConfig(fr, log_path)
            set_seed(eval_config.config["yaml-config"]['env']['seed'])
            eval_config.config["yaml-config"]['env']['wf_size'] = 'M'
            Builder(eval_config).build().eval()


if __name__ == "__main__":
    main()