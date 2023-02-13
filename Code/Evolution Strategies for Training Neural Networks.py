"""
Many practically valuable problems can be modeled as reinforcement learning problems. Control problems stand for an
important and widely studied family of reinforcement learning problems, including, for example, the cart pole problem,
the mountain car problem, the bipedal walker problem and the lunar lander problem. All these problems can be solved by
using policy neural networks trained with zeroth-order Evolution Strategies (ES) algorithms. The ZOO-RL Python library
supplemented with this project provides high-quality implementations of several zeroth-order ES algorithms, including
the famous OpenAI-ES algorithm. This task requires you to study the performance of OpenAI-ES using the ZOO-RL Python
library and report your findings.

- Study the benchmark cart pole problem named “CartPole-v1 ” and provide a short and clear description of this problem
    in your report. Particularly, define the state representation, the discrete action space, the reward function, the
    maximum length of any episode, and the criteria for successful learning. A short description of the CartPole-v1 problem
    can be found in ([https://gym.openai.com/envs/CartPole-v1/](https://gym.openai.com/envs/CartPole-v1/)). More information
    about the cart pole problem can be found in the reference below:

    G. Barto, R. S. Sutton and C. W. Anderson, “Neuronlike Adaptive Elements That Can
    Solve Difficult Learning Control Problem”, IEEE Transactions on Systems, Man, and
    Cybernetics, 1983.

- Experimentally evaluate the performance of OpenAI-ES at solving the CartPole-v1 problem. OpenAI-ES has already been
    implemented in ZOO-RL. No re-implementation of this algorithm is required for this task. All you need is to configure
    your YAML file requested by ZOO-RL to use this algorithm. Refer to README of the ZOO-RL Python library for more
    technical details. Report the architecture of the policy neural network used in your experiments. Provide the detailed
    algorithm parameter settings of OpenAI-ES.

- Draw the learning performance curves of OpenAI-ES with the horizontal axis representing the generation number and the
    vertical axis representing the performance of the trained policy neural network. You must perform 5 independent
    algorithm runs (each run must use a different random seed) on the CartPole-v1 problem and report the average learning
    performance in the respective performance curves.

"""
import os
import shutil
import subprocess

import time

import matplotlib.pyplot as plt
import pandas
from common import utils
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

zoo_path = os.path.abspath(os.path.join(os.getcwd(), "..", "Datasets", "zoo"))

if __name__ == '__main__':
    print("Running ZOO-RL...")

    for i in range(1, 6):
        print(f"Run {i} of 5")
        p = subprocess.Popen(["python", "run_rl.py", "--seed", str(i + 1), "--log" ], cwd=zoo_path)
        p.wait()
        print(f"{p.pid} finished")
    print("ZOO-RL finished")

    p = subprocess.Popen(["python", "eval_rl.py"], cwd=zoo_path)
    p.wait()
    print("eval status:", p.poll())

    log_path = os.path.abspath(os.path.join(zoo_path, "logs", "CartPole-v1"))
    print("Contents of logs folder:")
    logs = os.listdir(log_path)
    for i, log_folder in enumerate(logs):
        log_file = os.path.join(log_path, log_folder, "train_performance", "training_record.csv")
        print(log_folder)
        # read csv file
        try:
            pandas.read_csv(log_file)
            df = pandas.read_csv(log_file)
            #plot learning curve with row index as x axis and reward as y axis
            plt.plot(df.index+1, df.iloc[:, 1])
            plt.title(f"ZOO-RL (seed {i+1})")
            plt.xlabel("Generation")
            plt.ylabel("Reward")
            utils.output_plot(question_name=os.path.basename(__file__),
                              plot_name=f"ZOO-RL (seed {i+1})", plot=plt)
            plt.show()
        except FileNotFoundError:
            print(f"File not found: {log_file}")
        finally:
            plt.clf()
            plt.close()
            shutil.rmtree(os.path.join(log_path, log_folder))

    exit(0)

