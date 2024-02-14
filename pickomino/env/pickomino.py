import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

THROW_DICE = 0
ADD_TILE = 1
REMOVE_TILE = 2

class PickominoEnv(gym.Env):
    metadata = {"render_modes": None}

    def __init__(self, render_mode=None, nplayers=4):
        self.nplayers = nplayers
        self.player = 0
        self.number_of_dice = 8
        self.sides = 6
        self.number_of_tiles = 16
        self.first_tile = 21
        self.last_tile = 36
        self.rewards = {
            21: 1,
            22: 1,
            23: 1,
            24: 1,
            25: 2,
            26: 2,
            27: 2,
            28: 2,
            29: 3,
            30: 3,
            31: 3,
            32: 3,
            33: 4,
            34: 4,
            35: 4,
            36: 4,
        }

        self.reset()

        self.observation_space = spaces.Dict({
                "dice": spaces.MultiDiscrete(nvec=[self.sides]*self.number_of_dice, start=[1]*self.number_of_dice),
                "selected_dice": spaces.MultiBinary(n=self.number_of_dice),
                "tiles": self.MultiBinary(n=self.number_of_tiles),
                "player_stacks": [spaces.Sequence(spaces.Discrete(n=self.number_of_tiles, start=self.first_tile), stack=True) for _ in range(nplayers)],
            })

        # We have the following actions: throw dice, add tile to stack, put back tile
        self.action_space = spaces.Discrete(3)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
                "dice": self.dice,
                "selected_dice": self.selected_dice,
                "tiles": self.tiles,
                "player_stacks": self.player_stacks,
        }

    def _get_score(self):
        score = sum([self.rewards[tile] for tile in self.player_stacks[self.player]])
        return {
            "score": score,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.available_tiles = set(range(self.first_tile, self.last_tile+1))

        self.dice = np.ones(shape = (8,), dtype=np.int64)
        self.selected_dice = np.zeros(shape = (8,), dtype=np.int64)
        self.tiles = np.ones(shape = (16,), dtype=np.int64)
        self.player_stacks = [[] for _ in range(self.nplayers)]

        return self._get_obs, self._get_score

    def step(self, action):
        if action == THROW_DICE:
            mask = np.zeros(shape=(self.number_of_dice, self.sides), dtype=np.int8)
            sd = self.selected_dice.reshape((self.number_of_dice, 1))
            #dice that have been selected fix their value during sampling
            mask[np.argwhere(self.selected_dice != 0), sd[sd[:, 0] != 0]] = 1
            #all non-selected dice are rethrown and can assume any value
            mask[self.selected_dice==0,:] = 1
            mask = tuple([row for row in mask])
            self.dice = self.observation_space["dice"].sample(mask=mask)

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def close(self):
        pass