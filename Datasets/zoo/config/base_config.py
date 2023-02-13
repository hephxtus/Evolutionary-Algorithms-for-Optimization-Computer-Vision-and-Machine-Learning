from abc import *
import argparse
import yaml


class BaseConfig(metaclass=ABCMeta):

    def __init__(self, *args):
        parser = argparse.ArgumentParser(description='Arguments')
        parser.add_argument('--config', type=str,
                            default='config/cartpole_es_openai.yaml',
                            help='A config path for env, policy, and optim')
        parser.add_argument('--processor-num', type=int, default=2,
                            help='Specify processor number for multiprocessing')
        parser.add_argument('--eval-ep-num', type=int, default=1, help='Set evaluation number per iteration')
        # Settings related to logs
        parser.add_argument("--log", action="store_true", help="Use log")
        parser.add_argument('--save-model-freq', type=int, default=20, help='Save model every a few iterations')

        # Overwrite some common values in YAML with command-line options, if needed.
        parser.add_argument('--seed', type=int, default=None, help='Replace seed value in  YAML')

        args = parser.parse_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

            # Replace seed value if command-line options on seed is not None
            if args.seed is not None:
                config['env']['seed'] = args.seed

        self.config = {}
        self.config["runtime-config"] = vars(args)
        self.config["yaml-config"] = config
