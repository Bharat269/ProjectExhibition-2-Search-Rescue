import os
import numpy as np
import gym
from gym import spaces
import random
from datetime import datetime
import json

class CustomSearchAndRescueEnv(gym.Env):
    def __init__(self, config):
        super(CustomSearchAndRescueEnv, self).__init__()
        self.grid_size = config.get("grid_size", 10)
        self.num_agents = config.get("num_agents", 1)
        self.num_targets = config.get("num_targets", 1)
        self.observation_size = config.get("observation_size", 3)
        self.max_steps = config.get("max_steps", 100)
        self.output_file = config.get("output_file", "output.json")

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.observation_size, self.observation_size), dtype=np.float32)

        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.reset()

    def reset(self):
        self.time = 0
        self.agent_positions = {
            agent: (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            for agent in self.agents
        }
        self.agent_goals = {
            agent: (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            for agent in self.agents
        }
        self.target_positions = [
            (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            for _ in range(self.num_targets)
        ]
        self.visited_positions = set()
        self.agent_paths = {agent: [] for agent in self.agents}
        self.data_to_log = []

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations

    def step(self, actions):
        rewards = {}
        dones = {}
        infos = {}
        observations = {}

        for agent, action in actions.items():
            reward, done, info = self._update(agent, action)
            rewards[agent] = reward
            dones[agent] = done
            infos[agent] = info
            observations[agent] = self._get_observation(agent)

        self.time += 1
        done = self.time >= self.max_steps
        dones["__all__"] = done

        return observations, rewards, dones, infos

    def _update(self, agent, action):
        old_pos = self.agent_positions[agent]
        new_pos = self._move(old_pos, action)
        new_pos = tuple(int(i) for i in new_pos)
        self.agent_positions[agent] = new_pos
        self.visited_positions.add(new_pos)

        detected_targets = [t for t in self.target_positions if self._in_observation_range(new_pos, t)]
        reward = len(detected_targets) * 10.0

        agent_state = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent,
            "position": [float(x) for x in self._grid_to_gps(new_pos)],
            "goal": [float(x) for x in self._get_current_goal(agent)],
            "action": self._action_to_str(action),
            "step_number": int(self.time),
            "battery_level": float(max(0, 100 - (self.time * 0.5))),
            "orientation": self._get_orientation(action),
            "victim_found": bool(detected_targets),
            "needs_help": False,
            "reward": float(reward),
            "scan_confidence": float(0.8 - (len(self.visited_positions) * 0.01)),
            "surroundings": [[int(x), int(y)] for x, y in self._get_surroundings(new_pos)],
            "planned_path": [[float(a), float(b)] for a, b in self._generate_path(agent, self._get_current_goal(agent))]
        }

        self.data_to_log.append(agent_state)

        return reward, False, {}

    def _move(self, pos, action):
        x, y = pos
        if action == 0:    # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size - 1, y + 1)
        return (x, y)

    def _get_observation(self, agent):
        x, y = self.agent_positions[agent]
        half_size = self.observation_size // 2
        obs = np.zeros((self.observation_size, self.observation_size), dtype=np.float32)

        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                xi, yj = x + i, y + j
                if 0 <= xi < self.grid_size and 0 <= yj < self.grid_size:
                    obs[i + half_size, j + half_size] = 1.0

        return obs

    def _in_observation_range(self, pos, target):
        x, y = pos
        tx, ty = target
        return abs(x - tx) <= 1 and abs(y - ty) <= 1

    def _get_current_goal(self, agent):
        return tuple(int(i) for i in self.agent_goals[agent])

    def _grid_to_gps(self, pos):
        x, y = pos
        return [40.0 + float(x) * 0.001, -74.0 + float(y) * 0.001]

    def _action_to_str(self, action):
        return ["up", "down", "left", "right", "stay"][int(action)]

    def _get_orientation(self, action):
        return self._action_to_str(action)

    def _get_surroundings(self, pos):
        x, y = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        surroundings = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                surroundings.append((nx, ny))
        return surroundings

    def _generate_path(self, agent, goal):
        current_pos = self.agent_positions[agent]
        path = []

        x, y = current_pos
        gx, gy = goal

        while x != gx:
            x += 1 if gx > x else -1
            path.append((x, y))
        while y != gy:
            y += 1 if gy > y else -1
            path.append((x, y))

        return path

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        for pos in self.target_positions:
            grid[pos] = 'T'
        for agent, pos in self.agent_positions.items():
            grid[pos] = 'A'
        print("\n".join(" ".join(row) for row in grid))
        print("\n")

    def close(self):
        try:
            with open(self.output_file, "w") as f:
                json.dump(self.data_to_log, f, indent=2)
            print(f"Data written to {self.output_file}")
        except Exception as e:
            print(f"Error writing JSON: {e}")



config = {
    "grid_size": 10,
    "num_agents": 1,
    "num_targets": 2,
    "observation_size": 3,
    "max_steps": 50,
    "output_file": "output.json"
}

env = CustomSearchAndRescueEnv(config)

obs = env.reset()
done = {"__all__": False}
victim_found = False

while not done["__all__"] and not victim_found:
    actions = {agent: env.action_space.sample() for agent in env.agents}
    obs, rewards, done, infos = env.step(actions)
    env.render()

    for agent in env.agents:
        if infos.get(agent, {}).get("victim_found", False):
            print(f"\n🎯 Victim found by {agent} at step {infos[agent].get('step_number')}!")
            victim_found = True
            break

env.close()
