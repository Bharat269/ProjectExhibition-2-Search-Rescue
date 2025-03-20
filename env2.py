import pygame
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
import datetime  # For generating timestamps


class GridParallel:
    def __init__(self):
        pygame.init()
        self.sfm_params = {
            "desired_speed": 1.0,  # Desired speed of targets
            "repulsion_strength": 10.0,  # Strength of repulsion from obstacles/agents
            "personal_space_radius": 1.5,  # Radius for personal space
        }

        # Metrics tracking
        self.episode_returns = []  # Track cumulative rewards per episode
        self.success_count = 0  # Track successful episodes (all targets found)
        self.episode_lengths = []  # Track episode lengths (time steps)
        self.evaluation_interval = 100  # Evaluate every 100 episodes
        self.evaluation_returns = []  # Track evaluation performance

        self.target_velocities = {}
        self.time_step = 0.1

        # Grid size and full-screen setup
        self.grid_size = 6  # Larger grid
        self.num_obstacles = 3  # More obstacles
        self.cell_size = min(pygame.display.Info().current_w, pygame.display.Info().current_h) // self.grid_size
        self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size),
                                             pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.gamma = 0.9
        self.time = 1
        self.epsilon = .2  # to choose bw random action and qtable action, explore v exploit
        self.randomEnable = True  # if true, we can explore, otherwise exploit
        self.alpha = 0.4  # to stabilize updates. So new updates don't override old updates completely

        # Track visited positions
        self.visited_positions = {}  # {position: visit_count}

        # Agents (1 agents at different corners)
        self.agent_positions = {
            "rescue1": np.array([0, 0]),
            "rescue2": np.array([self.grid_size - 1, 0]),
        }

        # Randomly placed targets (2 targets)
        self.target_positions = set()
        while len(self.target_positions) < 2:
            target = tuple(np.random.randint(0, self.grid_size, size=2))
            if target not in {tuple(pos) for pos in self.agent_positions.values()}:  # preventing target agent overlap
                self.target_positions.add(target)

        # Obstacles placement
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:  # preventing target agent obst overlap
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos not in {tuple(pos) for pos in self.agent_positions.values()} and pos not in self.target_positions:
                self.obstacles.add(pos)

        self.actionspace = (1, 2, 3, 4)  # up down left right

        # Try to load existing qtable, otherwise create a new one
        self.qtable_file = "qtable1.pkl"
        if os.path.exists(self.qtable_file):
            try:
                with open(self.qtable_file, 'rb') as f:
                    self.qtable = pickle.load(f)
                print(f"Loaded Q-table from {self.qtable_file} with {len(self.qtable)} entries")
            except Exception as e:
                print(f"Error loading Q-table: {e}")
                self.qtable = dict()
        else:
            self.qtable = dict()  # dict {  ((obspos,selfpos),action):return }

    def target_sfm_forces(self, target_pos):
        # Driving force (random movement)
        driving_f = np.random.rand(2) * 2 - 1  # Random direction
        driving_f = driving_f / np.linalg.norm(driving_f + 1e-6) * self.sfm_params["desired_speed"]

        # Repulsive force from obstacles
        repulsive_f = self.repulsive_force(target_pos, self.obstacles)

        # Repulsive force from agents
        agent_positions = [pos for pos in self.agent_positions.values()]
        social_f = self.social_force(target_pos, agent_positions)

        # Total force
        total_force = driving_f + repulsive_f + social_f
        return total_force

    def repulsive_force(self, agent_pos, obstacles):
        """
        Calculate the repulsive force from obstacles.
        """
        force = np.zeros(2)
        for obs in obstacles:
            diff = agent_pos - obs
            distance = np.linalg.norm(diff)
            if distance < self.sfm_params["personal_space_radius"]:
                force += self.sfm_params["repulsion_strength"] * (diff / (distance ** 2 + 1e-6))
        return force

    def social_force(self, agent_pos, other_agents):
        """
        Calculate the social force from other agents.
        """
        force = np.zeros(2)
        for other_pos in other_agents:
            if np.array_equal(agent_pos, other_pos):
                continue  # Skip self
            diff = agent_pos - other_pos
            distance = np.linalg.norm(diff)
            if distance < self.sfm_params["personal_space_radius"]:
                force += self.sfm_params["repulsion_strength"] * (diff / (distance ** 2 + 1e-6))
        return force

    def update_targets(self):
        """
        Update target positions using SFM (Social Force Model).
        """
        for target in list(self.target_positions):
            target_pos = np.array(target)
            total_force = self.target_sfm_forces(target_pos)

            # Update velocity and position
            velocity = self.target_velocities.get(target, np.zeros(2))
            velocity += total_force * self.time_step
            new_pos = target_pos + velocity * self.time_step

            # Ensure the new position is within the grid
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)

            # Update target position and velocity
            self.target_positions.remove(target)
            self.target_positions.add(tuple(new_pos))
            self.target_velocities[tuple(new_pos)] = velocity

    def encode(x, y, grid_size):
        return x * grid_size + y

    def decode(index, grid_size):
        return index // grid_size, index % grid_size

    def update(self):  # Uses an actions dict {agent: action(0,1,2,3)}
        reward = 0
        episode_return = 0
        while True:
            self.update_targets()
            for agent in self.agent_positions:  # Iterate one agent at a time
                if not self.target_positions:
                    break
                cur = self.agent_positions[agent].copy()  # current position
                oldQ = tuple(GridParallel.encode(obs[0], obs[1], self.grid_size) for obs in self.obstacles)
                oldQ = tuple(list(oldQ) + [GridParallel.encode(cur[0], cur[1], self.grid_size)])
                for i in range(1, 5):  # adding missing values
                    if (oldQ, i) not in self.qtable:
                        self.qtable[(oldQ, i)] = 0
                # now choosing action according to exploration v exploitation
                action = -1
                randint = random.random()
                if randint < self.epsilon or self.qtable[(oldQ, 1)] == self.qtable[(oldQ, 2)] == self.qtable[
                    (oldQ, 3)] == self.qtable[(oldQ, 4)]:
                    action = np.random.choice([1, 2, 3, 4])
                else:
                    max_val = max(self.qtable[(oldQ, 1)], self.qtable[(oldQ, 2)], self.qtable[(oldQ, 3)],
                                  self.qtable[(oldQ, 4)])
                    for i in range(1, 5):
                        if self.qtable[(oldQ, i)] == max_val:
                            action = i
                            break

                reward = 0

                # Move agent
                new_pos = cur.copy()
                if action == 1:  # Up
                    new_pos[1] = cur[1] - 1
                elif action == 2:  # Down
                    new_pos[1] = cur[1] + 1
                elif action == 3:  # Left
                    new_pos[0] = cur[0] - 1
                elif action == 4:  # Right
                    new_pos[0] = cur[0] + 1

                new_vis = self.visibility(agent)  # Pass the agent name to visibility
                newQ = tuple(GridParallel.encode(obs[0], obs[1], self.grid_size) for obs in self.obstacles)
                newQ = tuple(list(newQ) + [GridParallel.encode(new_pos[0], new_pos[1], self.grid_size)])

                # Reward for moving closer to the nearest target
                min_old_dist = min(np.linalg.norm(cur - np.array(t)) for t in self.target_positions)
                min_new_dist = min(np.linalg.norm(new_pos - np.array(t)) for t in self.target_positions)
                if min_new_dist < min_old_dist:
                    reward += 3  # Moderate reward for getting closer

                # Check if the move is valid (within grid boundaries)
                if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                    reward = -5  # Increased penalty for hitting boundary
                elif tuple(new_pos) in self.obstacles:
                    reward = -10  # Increased penalty for hitting an obstacle
                else:
                    self.agent_positions[agent] = new_pos  # Update position only when valid

                    detected_targets = [t for t in self.target_positions if t in new_vis]
                    for target in detected_targets:
                        self.target_positions.remove(target)  # remove detected targets
                        reward = 50
                        break

                # Penalty for revisiting locations
                pos_tuple = tuple(new_pos)
                if pos_tuple in self.visited_positions:
                    self.visited_positions[pos_tuple] += 1
                    reward -= min(self.visited_positions[pos_tuple], 3)  # Slight penalty, capped at -3
                else:
                    self.visited_positions[pos_tuple] = 1

                if reward == 0:
                    reward = -2

                for i in range(1, 5):  # adding missing q vals if any
                    if (newQ, i) not in self.qtable.keys():
                        self.qtable[(newQ, i)] = 0

                td = (reward + self.gamma * max(self.qtable[(newQ, 1)], self.qtable[(newQ, 2)], self.qtable[(newQ, 3)],
                                               self.qtable[(newQ, 4)]) - self.qtable[(oldQ, action)])
                self.qtable[(oldQ, action)] = self.qtable[(oldQ, action)] + self.alpha * td
            self.time = self.time + 1

            episode_return += reward
            if not self.target_positions or self.time > 50:
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(self.time)
                if not self.target_positions:
                    self.success_count += 1
                break

    def reset(self):
        self.time = 0
        self.visited_positions.clear()  # Reset visited positions
        # Agents (1 agents at different corners)
        self.agent_positions = {
            "rescue1": np.array([0, 0]),
            "rescue2": np.array([self.grid_size - 1, 0]),
        }

        # Randomly placed targets (2 targets)
        self.target_positions = set()
        while len(self.target_positions) < 2:
            target = tuple(np.random.randint(0, self.grid_size, size=2))
            if target not in {tuple(pos) for pos in self.agent_positions.values()}:  # preventing target agent overlap
                self.target_positions.add(target)

        # Obstacles placement
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:  # preventing target agent obst overlap
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos not in {tuple(pos) for pos in self.agent_positions.values()} and pos not in self.target_positions:
                self.obstacles.add(pos)

    def visibility(self, agent):
        x, y = self.agent_positions[agent]  # Get agent's current position
        visible = []
        # Define relative movements for each direction (south, sw, w, nw, n, ne, e, se)
        directions = [
            (0, 1),  # South
            (-1, 1),  # South-West
            (-1, 0),  # West
            (-1, -1),  # North-West
            (0, -1),  # North
            (1, -1),  # North-East
            (1, 0),  # East
            (1, 1)  # South-East
        ]
        # Calculate visible positions while staying within grid boundaries
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                visible.append((new_x, new_y))
        return visible

    def render(self):
        self.screen.fill((255, 255, 255))

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 0),
                             (obs[0] * self.cell_size, obs[1] * self.cell_size, self.cell_size, self.cell_size))

        # Draw agents
        for agent, pos in self.agent_positions.items():
            pygame.draw.circle(self.screen, (0, 0, 255),
                               (pos[0] * self.cell_size + self.cell_size // 2, pos[1] * self.cell_size + self.cell_size // 2),
                               self.cell_size // 3)

        # Draw targets
        for target in self.target_positions:
            pygame.draw.circle(self.screen, (255, 0, 0),
                               (target[0] * self.cell_size + self.cell_size // 2, target[1] * self.cell_size + self.cell_size // 2),
                               self.cell_size // 3)
            # Draw velocity vector
            velocity = self.target_velocities.get(target, np.zeros(2))
            end_pos = (target[0] + velocity[0], target[1] + velocity[1])
            pygame.draw.line(self.screen, (255, 0, 0),
                             (target[0] * self.cell_size + self.cell_size // 2, target[1] * self.cell_size + self.cell_size // 2),
                             (end_pos[0] * self.cell_size + self.cell_size // 2, end_pos[1] * self.cell_size + self.cell_size // 2), 2)

        pygame.display.flip()

    def run(self):
        running = True
        fullscreen = True  # Start in fullscreen mode

        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False  # Press ESC to exit
                        elif event.key == pygame.K_f:
                            fullscreen = not fullscreen  # Toggle fullscreen
                            if fullscreen:
                                self.screen = pygame.display.set_mode(
                                    (self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.FULLSCREEN)
                            else:
                                self.screen = pygame.display.set_mode(
                                    (self.grid_size * self.cell_size, self.grid_size * self.cell_size))

                self.update()
                self.render()

                # Stop when all targets are found
                if not self.target_positions:
                    print("All targets found! Simulation complete.")
                    self.reset()
                    break
                if self.time > 50:
                    print("Time exceeded")
                    self.reset()
                    break
                self.clock.tick(10)
        finally:
            pygame.quit()  # Close pygame window
            if hasattr(self, 'evaluation_returns') and self.evaluation_returns:  # Check if evaluation_returns exists and is not empty
                self.plot_learning_curves()  # Plot learning curves after closing pygame
            self.close()

    def evaluate(self, num_episodes=10):
        """
        Evaluate the agent's performance over a number of episodes.
        """
        total_returns = []
        for _ in range(num_episodes):
            self.reset()
            episode_return = 0
            while True:
                reward = 0  # Initialize reward for each step
                self.update()
                episode_return += reward
                if not self.target_positions or self.time > 50:
                    break
            total_returns.append(episode_return)
        average_return = np.mean(total_returns)
        self.evaluation_returns.append(average_return)
        print(f"Evaluation: Average Return = {average_return}")

    def train(self, n):
        total_time = 0
        try:
            for episode in range(n):
                self.reset()
                while True:
                    self.update()
                    if not self.target_positions or self.time > 50:
                        total_time += self.time
                        break

                if (episode + 1) % self.evaluation_interval == 0:
                    self.evaluate()
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}: Average time = {total_time / (episode + 1)}")

            self.evaluate()
            success_rate = self.success_count / n
            average_episode_length = np.mean(self.episode_lengths)
            print(f"Training Complete: Success Rate = {success_rate}, Average Episode Length = {average_episode_length}")

        finally:
            self.close()  # Ensure proper cleanup when training ends

    def plot_learning_curves(self):
        """
        Plot the learning curves for episode returns and evaluation returns and save them as images.
        """
        if not self.episode_returns or not self.evaluation_returns:
            print("No data to plot. Ensure metrics are being tracked.")
            return

        plt.figure(figsize=(12, 6))

        # Plot episode returns
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_returns)
        plt.xlabel("Episode")
        plt.ylabel("Episode Return")
        plt.title("Episode Returns Over Time")

        # Plot evaluation returns
        plt.subplot(1, 2, 2)
        plt.plot(self.evaluation_returns)
        plt.xlabel("Evaluation Interval")
        plt.ylabel("Average Return")
        plt.title("Evaluation Returns Over Time")

        plt.tight_layout()

        # Save the plot as an image file
        plot_filename = "learning_curves.png"
        plt.savefig(plot_filename)
        print(f"Saved learning curves to {plot_filename}")

        # Optionally, close the plot to free up memory
        plt.close()

    def close(self):
        # Save the Q-table to a file and properly clean up resources.
        # Save Q-table
        try:
            with open(self.qtable_file, 'wb') as f:
                pickle.dump(self.qtable, f)
            print(f"Saved Q-table to {self.qtable_file} with {len(self.qtable)} entries")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

        # Clean up pygame resources
        pygame.quit()


# Run the simulation
env = GridParallel()
# env.randomEnable = False # uncomment to start exploiting
env.run()
env.train(1000)
env.close()