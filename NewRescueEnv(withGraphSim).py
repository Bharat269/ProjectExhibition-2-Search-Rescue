import numpy as np
import random
import matplotlib.pyplot as plt
from pettingzoo.utils.env import AECEnv
from gym import spaces

class RescueEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "rescue_v0"}
    OBSTACLE_COUNT = 3
    SURVIVOR_COUNT = 2  # Number of survivors

    def __init__(self, grid_size=8, max_steps=50, obst_count=OBSTACLE_COUNT, survivor_count=SURVIVOR_COUNT):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0

        self.OBSTACLE_COUNT = obst_count
        self.SURVIVOR_COUNT = survivor_count
        self.obs_pos = set()
        self.survivors_pos = set()

        # Randomly place obstacles
        while len(self.obs_pos) < self.OBSTACLE_COUNT:
            self.obs_pos.add((random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)))

        # Randomly place survivors
        while len(self.survivors_pos) < self.SURVIVOR_COUNT:
            self.survivors_pos.add((random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)))

        self.agents = ["agent_1", "agent_2"]
        self.pos = {}
        for agent in self.agents:
            while True:
                spawn = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if spawn not in self.obs_pos and spawn not in self.survivors_pos:  # Ensure agent doesn't spawn on an obstacle or survivor
                    self.pos[agent] = list(spawn)
                    break

        self.observation_spaces = {agent: spaces.Discrete(grid_size * grid_size) for agent in self.agents}
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}  # 4 directions
        self.render_mode = "human"
        self.found_survivors = {agent: False for agent in self.agents}  # Track whether agents have found survivors

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.obs_pos = set()
        self.survivors_pos = set()

        # Randomly place obstacles
        while len(self.obs_pos) < self.OBSTACLE_COUNT:
            self.obs_pos.add((random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)))

        # Randomly place survivors
        while len(self.survivors_pos) < self.SURVIVOR_COUNT:
            self.survivors_pos.add((random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)))

        self.pos = {}
        for agent in self.agents:
            while True:
                spawn = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                if spawn not in self.obs_pos and spawn not in self.survivors_pos:  # Ensure agent doesn't spawn on an obstacle or survivor
                    self.pos[agent] = list(spawn)
                    break

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.found_survivors = {agent: False for agent in self.agents}  # Reset survivors found status
        return {agent: self._get_obs(agent) for agent in self.agents}

    def step(self, actions):
        for agent, action in actions.items():
            if action == 0:  # UP
                new_pos = [self.pos[agent][0], max(0, self.pos[agent][1] - 1)]
            elif action == 1:  # DOWN
                new_pos = [self.pos[agent][0], min(self.grid_size - 1, self.pos[agent][1] + 1)]
            elif action == 2:  # LEFT
                new_pos = [max(0, self.pos[agent][0] - 1), self.pos[agent][1]]
            elif action == 3:  # RIGHT
                new_pos = [min(self.grid_size - 1, self.pos[agent][0] + 1), self.pos[agent][1]]

            # Ensure agent does not move onto an obstacle
            if tuple(new_pos) not in self.obs_pos:
                self.pos[agent] = new_pos

            # Check if agent found a survivor
            if tuple(self.pos[agent]) in self.survivors_pos and not self.found_survivors[agent]:
                self.found_survivors[agent] = True
                self.rewards[agent] = 10  # Reward for finding a survivor
            else:
                self.rewards[agent] = -1  # Small penalty for movement

        self.current_step += 1
        terminated = all(self.found_survivors.values()) or self.current_step >= self.max_steps
        self.terminations = {agent: terminated for agent in self.agents}

        # End the simulation if both agents have found their survivors
        if terminated:
            print("Both agents have found their survivors!")
        return {agent: self._get_obs(agent) for agent in self.agents}, self.rewards, self.terminations, {}

    def _get_obs(self, agent):
        return self.pos[agent][0] * self.grid_size + self.pos[agent][1]

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))

        # Mark obstacles on the grid
        for obs in self.obs_pos:
            grid[obs[1], obs[0]] = -1  # Mark obstacles with -1

        # Mark survivors on the grid
        for survivor in self.survivors_pos:
            grid[survivor[1], survivor[0]] = 2  # Mark survivors with 2

        # Mark agent positions on the grid
        for i, agent in enumerate(self.agents):
            x, y = self.pos[agent]
            grid[y, x] = i + 1  # Different values for each agent

        plt.imshow(grid, cmap="coolwarm", origin="upper")
        plt.xticks(range(self.grid_size))
        plt.yticks(range(self.grid_size))
        plt.grid(True)
        plt.show(block=False)
        plt.pause(0.5)
        plt.clf()

# Running the simulation
env = RescueEnv(grid_size=5)
env.reset()
print("Starting Multi-Agent Rescue Simulation...")
for _ in range(10):
    actions = {agent: np.random.choice(4) for agent in env.agents}
    obs, rewards, terminations, _ = env.step(actions)
    env.render()
    if all(terminations.values()):
        break  # End simulation if both agents have found their survivors
print('End')
