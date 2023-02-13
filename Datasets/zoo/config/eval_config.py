from abc import *
import argparse
import yaml
import os

# logging.basicConfig(level=logging.INFO)


class EvalConfig(metaclass=ABCMeta):

    def __init__(self, *args):
        fr, log_path = args
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str,
                            default=f"{log_path}/profile.yaml")
        parser.add_argument("--policy-path", type=str,
                            default=f'{log_path}/saved_models/ep_{fr}.pt',
                            help='saved model directory')
        parser.add_argument("--rms-path", type=str,
                            default=f'{log_path}/saved_models/ob_rms_{fr}.pickle',
                            help='saved run-time mean and std directory')
        parser.add_argument('--eval-ep-num', type=int, default=1, help='Set evaluation number per iteration')
        parser.add_argument('--save-model-freq', type=int, default=20, help='Save model every a few iterations')
        parser.add_argument('--processor-num', type=int, default=1, help='Testing model only use 1 processor')
        parser.add_argument("--log", action="store_true", help="Use log")
        parser.add_argument("--save-gif", action="store_true")

        args = parser.parse_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        if args.save_gif:
            run_num = args.ckpt_path.split("/")[-3]
            save_dir = f"test_gif/{run_num}/"
            os.makedirs(save_dir)

        self.config = {}
        self.config["runtime-config"] = vars(args)
        self.config["yaml-config"] = config

       
