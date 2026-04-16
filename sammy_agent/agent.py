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
    "PPO_Soccer_53fd9_00000_0_2026-04-16_00-11-22\checkpoint_000180\checkpoint-180",
)
POLICY_NAME = "default"  # this may be useful when training with selfplay


class SammyAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

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

        tune.registry.register_env("Soccer", lambda *_: BaseEnv())
        config["env"] = "Soccer"

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(CHECKPOINT_PATH)
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)
        print(f"Found policy with type: {type(self.policy)}")

        self.name = "Sammy"

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
        actions = {}
        for player_id in observation:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # since we only need the action, we discard the other elements
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions
