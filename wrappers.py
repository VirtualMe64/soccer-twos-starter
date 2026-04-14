import gym
from ray.rllib import MultiAgentEnv

from utils import RLLibWrapper

import soccer_twos

class InjectionWrapper(gym.core.Wrapper, MultiAgentEnv):

    def step(self, action):
        result = super().step(action)
        print("HI")
        return result

def create_rllib_env_with_wrapper(wrapper_cls=RLLibWrapper):
    return lambda config : _create_rllib_env_with_wrapper(config, wrapper_cls)

def _create_rllib_env_with_wrapper(env_config: dict = {}, wrapper_cls=RLLibWrapper):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return wrapper_cls(env)