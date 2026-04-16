import ray
from ray import tune
from soccer_twos import EnvType

from wrappers import *

NUM_ENVS_PER_WORKER = 3

if __name__ == "__main__":
    ray.init()

    create_rllib_env = create_rllib_env_with_wrapper(IndividualBallWrapper)
    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    def policy_mapping_fn(agent_id, **kwargs):
        return f"default" if int(agent_id) == 0 else f"agent_{agent_id}"

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_1",
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
            },
            "model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [
                256,
                256
                ],
                "vf_share_layers": True
            },
            "rollout_fragment_length": 5000  
        },
        stop={
            # "timesteps_total": 15000000,  # 15M
            # "time_total_s": 14400, # 4h
            "time_total_s": 3600 * 10, # 1h
        },
        checkpoint_freq=30,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="./ray_results/PPO_selfplay_1/hour2/checkpoint_000062/checkpoint-62",
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
