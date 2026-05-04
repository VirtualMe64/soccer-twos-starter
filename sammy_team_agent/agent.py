import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import create_rllib_env

ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    #PPO_Soccer_f41d3_00000_0_2026-04-14_19-37-52\checkpoint_000032\checkpoint-32",
    "injected_rewards_team\checkpoint_002900\checkpoint-2900" #2900 - 75%/70%, 2500 - 73%/71.5%, 1800- 74%/68%, 2929 - 73%/66.5%
) #Qualitatively - 2900 seems consistent, 1300 - 74-65%, 1600 - 74%-73.5%-69%

# against bert - 2200 - 54%

POLICY_NAME = "default"  # this may be useful when training with selfplay


class SammyTeamAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_gpus=0
        )

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")
        #config_path = "injected_rewards_team\checkpoint_000033\params.pkl"
        print(config_path, os.path.exists(config_path))
        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)

    
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0

        # create a dummy env since it's required but we only care about the policy
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"


        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(CHECKPOINT_PATH)
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)

        self.name = "Not Sammy Team Agent"

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        team_observations = np.concatenate(list(observation.values()), axis=0)
        team_actions, *_ = self.policy.compute_single_action(
            team_observations
        )
        actions = {}
        for player_id in observation:
            actions[player_id] = team_actions[int(player_id) * 3 : (int(player_id) + 1) * 3]
        return actions
