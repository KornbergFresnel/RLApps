import math
from math import pi
import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib.utils import merge_dicts
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def direction_to_coord(direction):
    """takes x \in [0,1), returns 2d coords on unit circle"""
    return tuple([math.cos(2 * pi * direction), math.sin(2 * pi * direction)])


def loss_game(dir1, dir2, starting_coords=(0, 0), alpha1=1, alpha2=1):
    coord1 = direction_to_coord(dir1)
    coord2 = direction_to_coord(dir2)
    x = (coord1[0] * alpha1 + coord2[0] * alpha2) + starting_coords[0]
    y = (coord1[1] * alpha1 + coord2[1] * alpha2) + starting_coords[1]
    val = math.sin(x) + math.sin(y)
    return val, x, y


DEFAULT_LOSS_GAME_CONFIG = {
    "alpha": 2.35,
    "total_moves": 2,
    "discrete_actions_for_players": [],
    "num_actions_per_dim": 100,
}


class LossGameAlphaMultiAgentEnv(MultiAgentEnv):
    def __init__(self, env_config):
        config = merge_dicts(DEFAULT_LOSS_GAME_CONFIG, env_config)
        self.alpha = config["alpha"]
        self.total_moves = config["total_moves"]
        self.discrete_actions_for_players = config["discrete_actions_for_players"]

        self.continuous_action_space = Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.num_actions_per_dim = config["num_actions_per_dim"]
        self.discrete_action_space = Discrete(n=self.num_actions_per_dim**2)

        for p in self.discrete_actions_for_players:
            if p not in [0, 1]:
                raise ValueError(
                    "Values for discrete_actions_for_players can only be [], [0], [1], or [0, 1]"
                )
        if len(self.discrete_actions_for_players) == 2:
            self.action_space = self.discrete_action_space
        else:
            self.action_space = self.continuous_action_space
        self.observation_space = Box(
            low=-500.0, high=500.0, shape=(3,), dtype=np.float32
        )
        self.player1 = 0
        self.player2 = 1
        self.num_moves = 0
        self.current_coord = [0, 0]
        self.obs = np.array([0, 0, 0])  # x, y, val

    def reset(self):
        self.num_moves = 0
        self.current_coord = [0, 0]
        self.obs = np.array([0, 0, 0])  # x, y, val
        return {
            self.player1: self.obs,
            self.player2: self.obs,
        }

    def _discrete_to_continuous_action(self, discrete_action):
        alpha_index = discrete_action % self.num_actions_per_dim
        alpha_percent = alpha_index / (self.num_actions_per_dim - 1)
        assert 0 - 1e-10 <= alpha_percent <= 1 + 1e-10

        direction_index = discrete_action // self.num_actions_per_dim
        direction_percent = direction_index / (self.num_actions_per_dim - 1)
        assert 0 - 1e-10 <= direction_percent <= 1 + 1e-10

        alpha_action = (
            alpha_percent
            * (
                self.continuous_action_space.high[1]
                - self.continuous_action_space.low[1]
            )
        ) + self.continuous_action_space.low[1]
        direction_action = (
            direction_percent
            * (
                self.continuous_action_space.high[0]
                - self.continuous_action_space.low[0]
            )
        ) + self.continuous_action_space.low[0]

        continuous_action = np.asarray(
            [direction_action, alpha_action], dtype=np.float32
        )

        assert (
            continuous_action in self.continuous_action_space
        ), f"continuous_action {continuous_action}, discrete_action {discrete_action}"
        return continuous_action

    def step(self, action_dict):
        move1 = action_dict[self.player1]
        move2 = action_dict[self.player2]

        if self.player1 in self.discrete_actions_for_players:
            move1 = self._discrete_to_continuous_action(discrete_action=move1)

        if self.player2 in self.discrete_actions_for_players:
            move2 = self._discrete_to_continuous_action(discrete_action=move2)

        assert move1 in self.continuous_action_space, f"move1: {move1}, move2: {move2}"
        assert move2 in self.continuous_action_space, f"move1: {move1}, move2: {move2}"

        move1 = (move1 + 1) / 2.0
        move2 = (move2 + 1) / 2.0

        move1, alpha1 = move1
        move2, alpha2 = move2

        assert 0 - 1e-10 <= move1 <= 1 + 1e-10
        assert 0 - 1e-10 <= move2 <= 1 + 1e-10

        assert 0 - 1e-10 <= alpha1 <= 1 + 1e-10
        assert 0 - 1e-10 <= alpha2 <= 1 + 1e-10

        alpha1 = alpha1 * self.alpha
        alpha2 = alpha2 * self.alpha

        val, x, y = loss_game(
            move1, move2, self.current_coord, alpha1=alpha1, alpha2=alpha2
        )
        self.obs = np.array([x, y, val])

        obs = {
            self.player1: self.obs,
            self.player2: self.obs,
        }

        self.current_coord = [x, y]
        r1 = val
        r2 = -val
        rew = {
            self.player1: r1,
            self.player2: r2,
        }
        self.num_moves += 1
        done = self.num_moves >= self.total_moves
        dones = {
            self.player1: done,
            self.player2: done,
            "__all__": done,
        }

        return obs, rew, dones, {}
