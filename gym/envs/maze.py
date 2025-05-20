import random
from itertools import chain
from typing import (
    Any,
    Final,
    Iterable,
    Literal,
    SupportsFloat,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray

from gym.core.agent import Agent, GridActions
from gym.core.constants import *
from gym.core.grid import Grid
from gym.core.object import Goal, Wall
from gym.core.world import GridWorld
from gym.multigrid import MultiGridEnv
from gym.typing import Position
from gym.utils.window import Window


class Maze(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        grid_type: int = 0,
        max_steps: int = 300,
        highlight_visible_cells: bool | None = False,
        tile_size: int = 10,
        state_representation: str = "positional",
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        ### fundamental parameters
        self.state_representation = state_representation

        if grid_type < 0 or grid_type >= 2:
            raise ValueError(
                f"The Fourroom only accepts grid_type of 0 and 1, given {grid_type}"
            )
        else:
            self.grid_type = grid_type

        self.max_steps = max_steps
        self.world = GridWorld
        self.actions_set = GridActions

        see_through_walls: bool = False

        self.agents = [
            Agent(
                self.world,
                color="blue",
                bg_color="light_blue",
                actions=self.actions_set,
                type="agent",
            )
        ]

        # Define positions for goals and agents
        self.goal_positions = [(17, 17)]
        self.agent_positions = [(7, 12)]

        # Explicit maze structure based on the image
        self.map = [
            "####################",
            "#                  #",
            "#                  #",
            "#       ###  ###  ##",
            "#       #      #   #",
            "#####          #   #",
            "#   #          #   #",
            "#   #   #      #   #",
            "#   #   #      #   #",
            "#   #   ########   #",
            "#   #      #   #   #",
            "#   #      #       #",
            "#          #       #",
            "#          #   #   #",
            "#   #      #   #####",
            "#   #      #       #",
            "#   ########       #",
            "#                  #",
            "#                  #",
            "####################",
        ]

        self.width = len(self.map[self.grid_type][0])
        self.height = len(self.map[self.grid_type])

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=self.agents,
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
            highlight_visible_cells=highlight_visible_cells,
            tile_size=tile_size,
        )

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        match self.state_representation:
            case "positional":
                observation_space = spaces.Box(
                    low=np.array([0, 0, 0, 0], dtype=np.float32),
                    high=np.array(
                        [self.width, self.height, self.width, self.height],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
            case "tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.width, self.height, self.world.encode_dim),
                    dtype=np.int64,
                )
            case _:
                raise ValueError(
                    f"Invalid state representation: {self.state_representation}"
                )

        return observation_space

    def _gen_grid(self, width, height, options):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        for x, row in enumerate(self.map):
            for y, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                elif cell == " ":
                    self.grid.set(x, y, None)

        # Place the goal
        goal = Goal(self.world, 0)
        self.put_obj(goal, *self.goal_positions[self.grid_type])
        goal.init_pos, goal.cur_pos = self.goal_positions[self.grid_type]

        # # place agent
        # if options["random_init_pos"]:
        #     coords = self.find_obj_coordinates(None)
        #     agent_positions = random.sample(coords, 1)[0]
        # else:
        agent_positions = self.agent_positions[self.grid_type]

        for agent in self.agents:
            self.place_agent(agent, pos=agent_positions)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        obs, info = super().reset(seed=seed, options=options)

        ### NOTE: not multiagent setting
        self.agent_pos = self.agents[0].pos

        ### NOTE: NOT MULTIAGENT SETTING
        observations = self.get_obs()
        info = {"success": False}

        return observations, info

    def step(self, actions):
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        actions = np.argmax(actions)
        actions = [actions]
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        info = {"success": False}

        for i in order:
            if (
                self.agents[i].terminated
                or self.agents[i].paused
                or not self.agents[i].started
            ):
                continue

            # Get the current agent position
            curr_pos = self.agents[i].pos
            done = False

            # Rotate left
            if actions[i] == self.actions.left:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, -1)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Rotate right
            elif actions[i] == self.actions.right:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, +1)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Move forward
            elif actions[i] == self.actions.up:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (-1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)

                # # Compute the distance between the current position and the goal
                # dist_reward = np.linalg.norm(
                #     np.array(self.goal_positions[self.grid_type]) - np.array(fwd_pos),
                #     ord=2,
                # )

                # # Normalize the distance with the maximum possible distance in the grid
                # dist_norm_reward = dist_reward / np.linalg.norm(
                #     [self.width, self.height], ord=2
                # )

                # # Invert the normalized distance to make reward larger as the agent gets closer
                # inverse_dist_reward = 1 - dist_norm_reward  # Closer => higher reward

                # # Scale the reward and add to the total rewards
                # rewards += (
                #     0.1 * inverse_dist_reward
                # )  # Weak reward signal, range 0 ~ 0.1

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif actions[i] == self.actions.down:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (+1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards = self._reward(i, rewards, 1)
                        info["success"] = True
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            elif actions[i] == self.actions.stay:
                # Get the contents of the cell in front of the agent
                fwd_pos = curr_pos
                fwd_cell = self.grid.get(*fwd_pos)
                self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                assert False, "unknown action"

        ### NOTE: not multiagent setting
        self.agent_pos = self.agents[0].pos

        terminated = done
        truncated = True if self.step_count >= self.max_steps else False

        observations = self.get_obs()

        return observations, rewards, terminated, truncated, info

    def get_obs(
        self,
    ):
        if self.state_representation == "positional":
            obs = np.array(
                [
                    self.agents[0].pos[0],
                    self.agents[0].pos[1],
                    self.goal_positions[self.grid_type][0],
                    self.goal_positions[self.grid_type][1],
                ]
            )
            obs = obs / np.maximum(self.grid_size[0], self.grid_size[1])
        elif self.state_representation == "tensor":
            obs = [
                self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
                for i in range(len(self.agents))
            ]
            obs = [self.world.normalize_obs * ob for ob in obs]
            obs = obs[0][:, :, 0:1]
        else:
            raise ValueError(
                f"Unknown state representation {self.state_representation}. "
                "Please use 'positional' or 'tensor'."
            )
        return obs
