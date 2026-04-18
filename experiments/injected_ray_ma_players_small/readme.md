# Approach
PPO self play training 4 separate models using injected rewards to try and help with sparse rewards of goals. Model itself is based on the baseline.

# Changes
Small model (single layer size 128 vs 2 layers size 256)
Tuned reward values (less emphasis on ball proximity to player, more emphasis on ball proximity to goal)

# Observations
Converges so so much faster

# Training Code
injected_ray_ma_players3.py
```python
import os
import ray
from ray import tune
from soccer_twos import EnvType

from wrappers import *

NUM_ENVS_PER_WORKER = 6
BASE_PORT = 8500


if __name__ == "__main__":
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    ray.init(include_dashboard=False)

    create_rllib_env = create_rllib_env_with_wrapper(IndividualBallWrapper)
    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player, "base_port": BASE_PORT})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    def policy_mapping_fn(agent_id, **kwargs):
        return f"default" if int(agent_id) == 0 else f"agent_{agent_id}"

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_3",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "agent_1": (None, obs_space, act_space, {}),
                    "agent_2": (None, obs_space, act_space, {}),
                    "agent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default", "agent_1", "agent_2", "agent_3"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
                "base_port": BASE_PORT,
            },
            "model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [
                128
                ],
                "vf_share_layers": True
            },
            "rollout_fragment_length": 5000  
        },
        stop={
            # "timesteps_total": 15000000,  # 15M
            # "time_total_s": 14400, # 4h
            "time_total_s": 3600 * 12, # 24h
        },
        checkpoint_freq=30,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_d2636_00000_0_2026-04-16_11-06-19/checkpoint_000637/checkpoint-637",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
```


```python
class IndividualBallWrapper(gym.core.Wrapper, MultiAgentEnv):
    def calculate_ball_position_reward(self, info, goal_x):
        MAX_REWARD = 0.0005
        MIN_REWARD = -0.0005
        MAX_DISTANCE = 30
        Y_FACTOR = 0.2

        ball_loc = info['ball_info']['position']
        dist = math.sqrt(((ball_loc[0] - goal_x) ** 2) + (Y_FACTOR * (ball_loc[1] ** 2)))
        base_reward = MAX_REWARD - (dist / MAX_DISTANCE) * (MAX_REWARD - MIN_REWARD)
        reward = max(MIN_REWARD, min(MAX_REWARD, base_reward))

        return reward
    
    def calculate_ball_proximity_reward(self, info):
        MAX_REWARD = 0.00025
        MIN_REWARD = -0.00025
        MAX_DISTANCE = 20

        ball_loc = info['ball_info']['position']
        player_loc = info['player_info']['position']
        dist = math.sqrt((ball_loc[0] - player_loc[0]) ** 2 + (ball_loc[1] - player_loc[1]) ** 2)
        base_reward = MAX_REWARD - (dist / MAX_DISTANCE) * (MAX_REWARD - MIN_REWARD)
        reward = max(MIN_REWARD, min(MAX_REWARD, base_reward))

        return reward

    def calculate_existence_reward(self):
        return -0.00005

    def step(self, action):
        observation, reward, done, info = super().step(action)

        for player_id in reward:
            goal_x = 15 if int(player_id) <= 1 else -15
            reward[player_id] += self.calculate_ball_position_reward(info[player_id], goal_x)
            reward[player_id] += self.calculate_ball_proximity_reward(info[player_id])
            reward[player_id] += self.calculate_existence_reward()

        return observation, reward, done, info
```