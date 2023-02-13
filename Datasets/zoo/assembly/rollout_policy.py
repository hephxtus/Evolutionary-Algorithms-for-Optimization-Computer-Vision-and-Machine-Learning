import numpy as np
import torch
import builder


def rollout_policy_worker(arguments):
    indi, env, optim, eval_ep_num, ob_rms_mean, ob_rms_std, processor_num, g, config = arguments

    if processor_num > 1:
        env = builder.build_env(config.config)

    hist_rewards = {}  # rewards record all evals
    hist_obs = {}  # observation  record all evals
    hist_actions = {}
    obs = None
    total_reward = 0

    for ep_num in range(eval_ep_num):
        # makesure identical training instances for each ep_num over one generation
        if ep_num == 0:
            states = env.reset(g)  # we also reset random.seed and np.random.seed in env.reset
        else:
            seed = np.random.randint(2 ** 31)  # same random seed across indi
            states = env.reset(seed)

        rewards_per_eval = []
        obs_per_eval = []
        actions_per_eval = []
        done = False

        for agent_id, model in indi.items():
            model.reset()
        while not done:
            actions = {}
            for agent_id, model in indi.items():
                s = states[agent_id]["state"]
                # reshape s
                if s.ndim < 2:  # make sure ndim of state = 2
                    s = s[np.newaxis, :]
                # update s
                if ob_rms_mean is not None:
                    s = (s - ob_rms_mean) / ob_rms_std
                # feed s into a model with respect to agent_id
                actions[agent_id] = model(s)
                obs_per_eval.append(s)
                actions_per_eval.append(actions[agent_id])
                # trace observations
                if obs is None:
                    obs = states[agent_id]["state"]
                else:
                    obs = np.append(obs, states[agent_id]["state"], axis=0)

            # feed actions of agents
            states, r, done, _ = env.step(actions)
            rewards_per_eval.append(r)
            total_reward += r

        hist_obs[ep_num] = obs_per_eval
        hist_actions[ep_num] = actions_per_eval
        hist_rewards[ep_num] = rewards_per_eval

    rewards_mean = total_reward / eval_ep_num
    # output based on different problems

    if ob_rms_mean is not None:
        return {'policy_id': indi['0'].policy_id, 'hist_obs': obs, 'rewards': rewards_mean}
    else:
        return {'policy_id': indi['0'].policy_id,
                'rewards': rewards_mean}


def discount_rewards(rewards):
    gamma = 0.99  # gamma: discount factor in rl
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards
