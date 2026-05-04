import gym
from ray.rllib import MultiAgentEnv, BaseEnv
import soccer_twos

import math

from utils import RLLibWrapper

class TeamBallWrapper(gym.core.Wrapper, MultiAgentEnv):
    def calculate_ball_position_reward(self, info):
        MAX_REWARD = 0.005 #0.05
        MIN_REWARD = -0.005 #-0.05
        MAX_DISTANCE = 30
        HALF_DISTANCE = MAX_DISTANCE / 2

        ball_loc = info[0][0]['ball_info']['position']
        # team 0 is on the left, wants ball to be close to (15, 0)
        # team 1 is on the right, wants ball to be close to (-15, 0)
        # distances are capped at 30, 0 to 30 lerps to MAX_REWARD to MIN_REWARD

        reward = {}
        dist_team_0 = math.sqrt((ball_loc[0] - HALF_DISTANCE) ** 2 + ball_loc[1] ** 2)
        dist_team_1 = math.sqrt((ball_loc[0] + HALF_DISTANCE) ** 2 + ball_loc[1] ** 2)
        base_reward_team_0 = MAX_REWARD - (dist_team_0 / MAX_DISTANCE) * (MAX_REWARD - MIN_REWARD)
        base_reward_team_1 = MAX_REWARD - (dist_team_1 / MAX_DISTANCE) * (MAX_REWARD - MIN_REWARD)
        reward[0] = max(MIN_REWARD, min(MAX_REWARD, base_reward_team_0))
        reward[1] = max(MIN_REWARD, min(MAX_REWARD, base_reward_team_1))

        return reward
    
    def calculate_ball_proximity_reward(self, info):
        MAX_REWARD = 0.001 #0.05
        MIN_REWARD = -0.001 #-0.05
        MAX_DISTANCE = 20
        
        ball_loc = info[0][0]['ball_info']['position']
        reward = {}
        for team_id in [0, 1]:
            for player_id in [0, 1]:
                player_loc = info[team_id][player_id]['player_info']['position']
                dist = math.sqrt((ball_loc[0] - player_loc[0]) ** 2 + (ball_loc[1] - player_loc[1]) ** 2)
                base_reward = MAX_REWARD - (dist / MAX_DISTANCE) * (MAX_REWARD - MIN_REWARD)
                reward[team_id] = reward.get(team_id, 0) + max(MIN_REWARD, min(MAX_REWARD, base_reward))

        return reward

    def calculate_kick_direction_reward(self, info):
        MAX_REWARD = 0.005 #0.05

        reward = {}

        ball_pos = info[0][0]['ball_info']['position']
        ball_vel = info[0][0]['ball_info']['velocity']

        for team_id in [0, 1]:
            # goal positions
            goal_x = 15 if team_id == 0 else -15
            goal_vec = [goal_x - ball_pos[0], -ball_pos[1]]

            # normalize
            goal_norm = math.sqrt(goal_vec[0]**2 + goal_vec[1]**2) + 1e-8
            vel_norm = math.sqrt(ball_vel[0]**2 + ball_vel[1]**2) + 1e-8

            goal_dir = [goal_vec[0]/goal_norm, goal_vec[1]/goal_norm]
            vel_dir = [ball_vel[0]/vel_norm, ball_vel[1]/vel_norm]

            # dot product = alignment
            alignment = goal_dir[0]*vel_dir[0] + goal_dir[1]*vel_dir[1]

            reward[team_id] = MAX_REWARD * alignment  # [-MAX, MAX]

        return reward

    def calculate_kick_speed_reward(self, info):
        MAX_REWARD = 0.001
        MAX_SPEED = 20 #may need to change

        ball_vel = info[0][0]['ball_info']['velocity']
        speed = math.sqrt(ball_vel[0]**2 + ball_vel[1]**2)

        scaled = min(speed / MAX_SPEED, 1.0)

        return {
            0: MAX_REWARD * scaled,
            1: MAX_REWARD * scaled
        }

    def step(self, action):
        observation, reward, done, info = super().step(action)
        
        ball_pos_rewards = self.calculate_ball_position_reward(info)
        ball_proximity_rewards = self.calculate_ball_proximity_reward(info)
        direction_rewards = self.calculate_kick_direction_reward(info)
        speed_rewards = self.calculate_kick_speed_reward(info)

        for team_id in reward:
            reward[team_id] += ball_pos_rewards[team_id]
            reward[team_id] += ball_proximity_rewards[team_id]
            reward[team_id] += direction_rewards[team_id]
            reward[team_id] += speed_rewards[team_id]

        return observation, reward, done, info

class IndividualBallWrapper(gym.core.Wrapper, MultiAgentEnv):
    def calculate_ball_position_reward(self, info, goal_x):
        MAX_REWARD = 0.0001
        MIN_REWARD = -0.0001
        MAX_DISTANCE = 30
        Y_FACTOR = 0.2

        ball_loc = info['ball_info']['position']
        dist = math.sqrt(((ball_loc[0] - goal_x) ** 2) + (Y_FACTOR * (ball_loc[1] ** 2)))
        base_reward = MAX_REWARD - (dist / MAX_DISTANCE) * (MAX_REWARD - MIN_REWARD)
        reward = max(MIN_REWARD, min(MAX_REWARD, base_reward))

        return reward
    
    def calculate_ball_proximity_reward(self, info):
        MAX_REWARD = 0.0005
        MIN_REWARD = 0
        MAX_DISTANCE = 20

        ball_loc = info['ball_info']['position']
        player_loc = info['player_info']['position']
        dist = math.sqrt((ball_loc[0] - player_loc[0]) ** 2 + (ball_loc[1] - player_loc[1]) ** 2)
        base_reward = MAX_REWARD - (dist / MAX_DISTANCE) * (MAX_REWARD - MIN_REWARD)
        reward = max(MIN_REWARD, min(MAX_REWARD, base_reward))

        return reward

    def calculate_existence_reward(self, info):
        return -0.00005

    def step(self, action):
        observation, reward, done, info = super().step(action)

        for player_id in reward:
            goal_x = 15 if int(player_id) <= 1 else -15
            reward[player_id] += self.calculate_ball_position_reward(info[player_id], goal_x)
            reward[player_id] += self.calculate_ball_proximity_reward(info[player_id])
            reward[player_id] += self.calculate_existence_reward(info)

        return observation, reward, done, info

class MoveRightWrapper(gym.core.Wrapper):
    def calculate_x_pos_reward(self, info):
        player_loc = info['player_info']['position']
        player_x = player_loc[0]

        return player_x

    def step(self, action):
        observation, reward, done, info = super().step(action)

        reward += self.calculate_x_pos_reward(info)

        return observation, reward, done, info

def create_rllib_env_with_wrapper(wrapper_cls=RLLibWrapper):
    return lambda config : _create_rllib_env_with_wrapper(config, wrapper_cls)

def _create_rllib_env_with_wrapper(env_config: dict = {}, wrapper_cls = None):
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
        return env if wrapper_cls is None else wrapper_cls(env)
    return env if wrapper_cls is None else wrapper_cls(env)