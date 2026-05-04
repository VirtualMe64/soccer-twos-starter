import ray
from ray import tune
from soccer_twos import EnvType

from wrappers import *

NUM_ENVS_PER_WORKER = 2


if __name__ == "__main__":
    ray.init()

    create_rllib_env = create_rllib_env_with_wrapper(TeamBallWrapper)
    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_team})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_teams_6",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 12,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "gamma": 0.9, #may need to change this (0.9), default is 0.99
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(lambda _: "default"),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_team,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [128], #256
            },
        },
        stop={
            # "timesteps_total": 15000000,  # 15M
            # "timesteps_total": 150000 # 150k
            "time_total_s": 864000
            #"time_total_s": 1200, # 20m
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="./ray_results/PPO_teams_6/PPO_Soccer_6e539_00000_0_2026-04-20_22-40-17/checkpoint_001200/checkpoint-1200",
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
