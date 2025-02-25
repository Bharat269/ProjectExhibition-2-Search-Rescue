import pygame
import numpy as np
import random

class GridParallel:
    def __init__(self):
        pygame.init()

        # Grid size and full-screen setup
        self.grid_size = 6  # Larger grid
        self.num_obstacles = 3  # More obstacles
        self.cell_size = min(pygame.display.Info().current_w, pygame.display.Info().current_h) // self.grid_size
        self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.gamma = 0.9
        self.time = 1
        self.epsilon = .2 # to choose bw random action and qtable action, explore v exploit
        self.randomEnable = True # if true, we can explore, otherwise exploit
        self.alpha = 0.4 # to stabilize updates. So new updates don't override old updates completely
        
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
            if target not in {tuple(pos) for pos in self.agent_positions.values()}:    # preventing target agent overlap
                self.target_positions.add(target)

        # Obstacles placement
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:                                     # preventing target agent obst overlap
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos not in {tuple(pos) for pos in self.agent_positions.values()} and pos not in self.target_positions:
                self.obstacles.add(pos)

        self.actionspace = (1, 2, 3, 4)  # up down left right
        self.qtable = dict()  # dict {  ((obspos,selfpos),action):return }
    
    def encode(x, y, grid_size):
        return x * grid_size + y

    def decode(index, grid_size):
        return index // grid_size, index % grid_size
    
    def update(self):  # Uses an actions dict {agent: action(0,1,2,3)}
        for agent in self.agent_positions:  # Iterate one agent at a time
            if not self.target_positions:
                break
            cur = self.agent_positions[agent].copy()  # current position
            oldQ = tuple(GridParallel.encode(obs[0],obs[1],self.grid_size) for obs in self.obstacles)
            oldQ = tuple(list(oldQ) + [GridParallel.encode(cur[0],cur[1],self.grid_size)])
            for i in range(1, 5):   # adding missing values
                if (oldQ, i) not in self.qtable:
                    self.qtable[(oldQ, i)] = 0
            # now choosing action according to exploration v exploitation
            action = -1
            randint = random.random()
            if randint < self.epsilon or self.qtable[(oldQ, 1)] == self.qtable[(oldQ, 2)] == self.qtable[(oldQ, 3)] == self.qtable[(oldQ, 4)]:
                action = np.random.choice([1, 2, 3, 4])
            else:
                max_val = max(self.qtable[(oldQ, 1)], self.qtable[(oldQ, 2)], self.qtable[(oldQ, 3)], self.qtable[(oldQ, 4)])
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
            newQ = tuple(GridParallel.encode(obs[0],obs[1],self.grid_size) for obs in self.obstacles)
            newQ = tuple(list(newQ) + [GridParallel.encode(new_pos[0],new_pos[1],self.grid_size)])
            
            # Reward for moving closer to the nearest target
            min_old_dist = min(np.linalg.norm(cur - np.array(t)) for t in self.target_positions)
            min_new_dist = min(np.linalg.norm(new_pos - np.array(t)) for t in self.target_positions)
            if min_new_dist < min_old_dist:
                reward += 3  # Moderate reward for getting closer
            
            # Check if the move is valid (within grid boundaries)
            if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                reward = -5  # Increased penalty for hitting boundary
                #print(f"{agent} hit boundary at {new_pos}")
            elif tuple(new_pos) in self.obstacles:
                reward = -10  # Increased penalty for hitting an obstacle
                #print(f"{agent} hit an obstacle at {new_pos}, reward = {reward}")
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
            
            for i in range(1, 5):   # adding missing q vals if any
                if (newQ, i) not in self.qtable.keys():
                    self.qtable[(newQ, i)] = 0
            
            td = (reward + self.gamma * max(self.qtable[(newQ, 1)], self.qtable[(newQ, 2)], self.qtable[(newQ, 3)], self.qtable[(newQ, 4)]) - self.qtable[(oldQ, action)])
            self.qtable[(oldQ, action)] = self.qtable[(oldQ, action)] + self.alpha * td
        self.time = self.time + 1
    
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
            if target not in {tuple(pos) for pos in self.agent_positions.values()}:    # preventing target agent overlap
                self.target_positions.add(target)

        # Obstacles placement
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:                                     # preventing target agent obst overlap
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos not in {tuple(pos) for pos in self.agent_positions.values()} and pos not in self.target_positions:
                self.obstacles.add(pos)

    def visibility(self, agent):
        x, y = self.agent_positions[agent]  # Get agent's current position
        visible = []
        # Define relative movements for each direction (south, sw, w, nw, n, ne, e, se)
        directions = [
            (0, 1),   # South
            (-1, 1),  # South-West
            (-1, 0),  # West
            (-1, -1), # North-West
            (0, -1),  # North
            (1, -1),  # North-East
            (1, 0),   # East
            (1, 1)    # South-East
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
            pygame.draw.rect(self.screen, (0, 0, 0), (obs[0] * self.cell_size, obs[1] * self.cell_size, self.cell_size, self.cell_size))

        # Draw agents
        for agent, pos in self.agent_positions.items():
            pygame.draw.circle(self.screen, (0, 0, 255), (pos[0] * self.cell_size + self.cell_size // 2, pos[1] * self.cell_size + self.cell_size // 2), self.cell_size // 3)

        # Draw targets
        for target in self.target_positions:
            pygame.draw.circle(self.screen, (255, 0, 0), (target[0] * self.cell_size + self.cell_size // 2, target[1] * self.cell_size + self.cell_size // 2), self.cell_size // 3)

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
                                self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.FULLSCREEN)
                            else:
                                self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))

                self.update()
                self.render()

                # Stop when all targets are found
                if not self.target_positions:
                    print("All targets found! Simulation complete.")
                    env.reset()
                    break
                if self.time>50:
                    print("Time exceeded")
                    env.reset()
                    break
                self.clock.tick(10)
        finally:
            pygame.quit()  # Ensure pygame quits when the loop ends

    def train(self,n):
        total = 0
        for _ in range(n):
            while True:
                env.update()
                if not self.target_positions or self.time>40:
                    total = total + self.time
                    env.reset()
                    break
        print(f"Average time in run: {total/n}")

# Run the simulation
env = GridParallel()
# env.randomEnable = False # uncomment to start exploiting
env.run()
