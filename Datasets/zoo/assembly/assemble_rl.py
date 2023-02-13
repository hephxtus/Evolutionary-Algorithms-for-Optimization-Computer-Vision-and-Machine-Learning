import multiprocessing as mp
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
# import pickle5 as pickle
import yaml

from assembly.base_assemble import BaseAssembleRL
from assembly.rollout_policy import rollout_policy_worker
from utils.running_mean_std import RunningMeanStd
from env.gym_openAI.simulator_gym import GymEnv

from utils.policy_dict import agent_policy

class AssembleRL(BaseAssembleRL):

    def __init__(self, config, env, policy, optim):
        super(AssembleRL, self).__init__()

        self.config = config

        self.env = env
        self.policy = policy
        self.optim = optim

        #  settings for running
        self.running_mstd = self.config.config['yaml-config']["optim"]['input_running_mean_std']
        if self.running_mstd:  # Init running mean and std
            if isinstance(self.env, GymEnv):
                self.ob_rms = RunningMeanStd(shape=self.env.env.observation_space.shape)
            else:
                self.ob_rms = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean = self.ob_rms.mean
            self.ob_rms_std = np.sqrt(self.ob_rms.var)
        else:
            self.ob_rms = None
            self.ob_rms_mean = None
            self.ob_rms_std = None

        self.generation_num = self.config.config['yaml-config']["optim"]['generation_num']
        self.processor_num = self.config.config['runtime-config']['processor_num']
        self.eval_ep_num = self.config.config['runtime-config']['eval_ep_num']

        # log settings
        self.log = self.config.config['runtime-config']['log']
        self.save_model_freq = self.config.config['runtime-config']['save_model_freq']
        self.save_mode_dir = None

    def train(self):
        if self.log:
            # Init log repository
            now = datetime.now()
            curr_time = now.strftime("%Y%m%d%H%M%S%f")
            dir_lst = []
            self.save_mode_dir = f"logs/{self.env.name}/{curr_time}"
            dir_lst.append(self.save_mode_dir)
            dir_lst.append(self.save_mode_dir + "/saved_models/")
            dir_lst.append(self.save_mode_dir + "/train_performance/")
            for _dir in dir_lst:
                os.makedirs(_dir)
            # shutil.copyfile(self.args.config, self.save_mode_dir + "/profile.yaml")
            # save the running YAML as profile.yaml in the log
            with open(self.save_mode_dir + "/profile.yaml", 'w') as file:
                yaml.dump(self.config.config['yaml-config'], file)
                file.close()

        # Start with a population init
        population = self.optim.init_population(self.policy, self.env)

        if self.config.config['yaml-config']['optim']['maximization']:
            best_reward_so_far = float("-inf")
        else:
            best_reward_so_far = float("inf")

        for g in range(self.generation_num):
            start_time = time.time()

            # start multiprocessing
            p = mp.Pool(self.processor_num)

            arguments = [(indi, self.env, self.optim, self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                          self.processor_num, g, self.config) for indi in population]

            # start rollout works
            start_time_rollout = time.time()

            if self.processor_num > 1:
                results = p.map(rollout_policy_worker, arguments)
            else:
                results = [rollout_policy_worker(arg) for arg in arguments]

            p.close()

            # end rollout
            end_time_rollout = time.time() - start_time_rollout

            # start eval
            start_time_eval = time.time()
            results_df = pd.DataFrame(results).sort_values(by=['policy_id'])

            population, sigma_curr, best_reward_per_g = self.optim.next_population(self, results_df)
            end_time_eval = time.time() - start_time_eval

            end_time_generation = time.time() - start_time

            # update best reward so far
            if self.config.config['yaml-config']['optim']['maximization'] and (best_reward_per_g > best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            if (not self.config.config['yaml-config']['optim']['maximization']) and (
                    best_reward_per_g < best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            # print runtime infor
            print(
                f"episode: {g}, best reward so far: {best_reward_so_far:.4f}, best reward of the current generation: {best_reward_per_g:.4f}, sigma: {sigma_curr:.3f}, time_generation: {end_time_generation:.2f}, rollout_time: {end_time_rollout:.2f}, eval_time: {end_time_eval:.2f}",
                flush=True
            )

            # update mean and std every generation
            if self.running_mstd:
                hist_obs = []
                hist_obs = np.concatenate(results_df['hist_obs'])
                # Update future ob_rms_mean  and  ob_rms_std
                self.ob_rms.update(hist_obs)
                self.ob_rms_mean = self.ob_rms.mean
                self.ob_rms_std = np.sqrt(self.ob_rms.var)

            if self.log:
                if self.running_mstd:
                    results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from  log
                # return row of parent policy, i.e., policy_id = -1
                results_df = results_df.loc[results_df['policy_id'] == -1]
                with open(self.save_mode_dir + "/train_performance" + "/training_record.csv", "a") as f:
                    results_df.to_csv(f, index=False, header=False)

                elite = self.optim.get_elite_model()
                if (g + 1) % self.save_model_freq == 0:
                    save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{(g + 1)}.pt"
                    torch.save(elite.state_dict(), save_pth)
                    if self.running_mstd:
                        save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{(g + 1)}.pickle"
                        f = open(save_pth, 'wb')
                        pickle.dump(np.concatenate((self.ob_rms_mean, self.ob_rms_std)), f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()

    def eval(self):
        # load policy from log
        self.policy.load_state_dict(torch.load(self.config.config['runtime-config']['policy_path']))
        # create an individual wrapped with agent id
        indi = agent_policy(self.env.get_agent_ids(), self.policy)
        # load runtime mean and std
        if self.running_mstd:
            with open(self.config.config['runtime-config']['rms_path'], "rb") as f:
                ob_rms = pickle.load(f)
                self.ob_rms_mean = ob_rms[:int(0.5 * len(ob_rms))]
                self.ob_rms_std = ob_rms[int(0.5 * len(ob_rms)):]

        self.policy.eval()
        # use a random seed for simulator in testing setting
        g = np.random.randint(2 ** 31)

        arguments = [(indi, self.env, self.optim, self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                      self.processor_num, g, self.config)]

        results = [rollout_policy_worker(arg) for arg in arguments]

        results_df = pd.DataFrame(results, columns=['policy_id', 'reward', 'hist_obs', 'hist_act', 'hist_rew', 'hist_done'])

        if self.log:
            if 'hist_obs' in results_df.columns:
                results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from  log
            dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance"
            if not os.path.exists(dir_test):
                os.makedirs(dir_test)
            results_df.to_csv(dir_test + "/testing_record.csv", index=False, header=False, mode='a')