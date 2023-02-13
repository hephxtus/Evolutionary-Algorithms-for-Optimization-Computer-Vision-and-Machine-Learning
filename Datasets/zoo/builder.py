from assembly.assemble_rl import AssembleRL
from utils.utils import get_state_num, get_action_num, is_discrete_action, get_nn_output_num


class Builder:
    def __init__(self, baseconfig):
        self.config = baseconfig
        self.env = None
        self.policy = None
        self.optim = None
        self.shared_params = None

    def build(self):
        # build environment
        env = build_env(self.config.config)

        # build one policy or more
        policy_number = 0
        policy_dict = {}
        self.shared_params = self.config.config['yaml-config']['policy']['shared_params']

        for agent_id in self.config.config['yaml-config']['policy']:
            if agent_id != 'shared_params':
                # based on the environment, decide if the action space is discrete
                self.config.config['yaml-config']['policy'][agent_id]["discrete_action"] = is_discrete_action(env,
                                                                                                              agent_id)
                # based on the environment, generate the state num to build policy
                self.config.config['yaml-config']['policy'][agent_id]["state_num"] = get_state_num(env, agent_id)
                # based on the environment, generate the action num to build policy
                self.config.config['yaml-config']['policy'][agent_id]["action_num"] = get_nn_output_num(env, agent_id)
                policy_dict[agent_id] = build_policy(self.config.config['yaml-config']["policy"][agent_id])
                policy_number += 1

                if self.shared_params:
                    break

        # build optimizer
        optim = build_optim(self.config.config['yaml-config']["optim"])

        if policy_number == 1:
            # single-policy for single agent/shared-params multi-agent always use agent_0
            return AssembleRL(self.config, env, policy_dict['agent_0'], optim)


def build_env(config):
    env_name = config['yaml-config']["env"]["name"]
    if env_name in ["LunarLanderContinuous-v2", "CartPole-v1"]:
        from env.gym_openAI.simulator_gym import GymEnv
        return GymEnv(env_name, config['yaml-config']["env"])
    else:
        raise AssertionError(f"{env_name} doesn't support, please specify supported a env in yaml.")


def build_policy(config):
    model_name = config["name"]
    if model_name == "model_rnn":
        from policy.gym_model import GymPolicy
        return GymPolicy(config)
    else:
        raise AssertionError(f"{model_name} doesn't support, please specify supported a model in yaml.")


def build_optim(config):
    optim_name = config["name"]
    if optim_name == "es_openai":
        from optim.es.es_openai import ESOpenAI
        return ESOpenAI(config)
    else:
        raise AssertionError(f"{optim_name} doesn't support, please specify supported a optim in yaml.")
