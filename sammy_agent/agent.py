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
    "../experiments/injected_ray_ma_players/hours13-24/"
    "checkpoint_001426/checkpoint-1426",
)
print(CHECKPOINT_PATH)
POLICY_NAMES = ["default", "agent_1", "agent_2", "agent_3"]  # this may be useful when training with selfplay

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
        self.policies = []
        for policy_name in POLICY_NAMES:
            cls = get_trainable_cls(ALGORITHM)
            agent = cls(env=config["env"], config=config)
            # load state from checkpoint
            agent.restore(CHECKPOINT_PATH)
            # get policy for evaluation
            policy = agent.get_policy(policy_name)
            print(f"Found policy with type: {type(policy)}")
            self.policies.append(policy)

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
        for i, player_id in enumerate(observation):
            # compute_single_action returns a tuple of (action, action_info, ...)
            # since we only need the action, we discard the other elements
            possible_actions = []
            for policy in self.policies:
                action, *_ = policy.compute_single_action(observation[player_id])
                possible_actions.append(action)
            # for each of the 3 entries, we take the majority vote among the policies
            # if a tie, choose randomly among the tied actions
            final_action = []
            for j in range(3):
                counts = {}
                for action in possible_actions:
                    a = action[j]
                    if a not in counts:
                        counts[a] = 0
                    counts[a] += 1
                max_count = max(counts.values())
                candidates = [a for a, count in counts.items() if count == max_count]
                final_action.append(np.random.choice(candidates))
            actions[player_id] = np.array(final_action)
            
        return actions
