import pygame
import numpy as np
import random

class GridParallel:
    def __init__(self):
        pygame.init()

        # Grid size and full-screen setup
        self.grid_size = 5  # Larger grid
        self.num_obstacles = 3  # More obstacles
        self.cell_size = min(pygame.display.Info().current_w, pygame.display.Info().current_h) // self.grid_size
        self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.gamma = 0.9
        self.time = 1
        self.epsilon = .2 # to choose bw random action and qtable action, explore v exploit
        self.randomEnable = True # if true, we can explore, otherwise exploit
        # Agents (1 agents at different corners)
        self.agent_positions = {
            "rescue1": np.array([0, 0]),
            # "rescue2": np.array([self.grid_size - 1, 0]),
            # "rescue3": np.array([0, self.grid_size - 1]),
            # "rescue4": np.array([self.grid_size - 1, self.grid_size - 1])
        }

        # Randomly placed targets (2 targets)
        self.target_positions = set()
        while len(self.target_positions) < 2:
            target = tuple(np.random.randint(0, self.grid_size, size=2))
            if target not in {tuple(pos) for pos in self.agent_positions.values()}:    #preventing target agent overlap
                self.target_positions.add(target)

        # Obstacles placement
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:                                     #preventing target agent obst overlap
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos not in {tuple(pos) for pos in self.agent_positions.values()} and pos not in self.target_positions:
                self.obstacles.add(pos)

        self.actionspace = (1,2,3,4) # up down left right
        self.qtable = dict() #dict {  (visibility,action):return }

    def update(self):  # Uses an actions dict {agent: action(0,1,2,3)}
        for agent in self.agent_positions:  # Iterate one agent at a time
            
            cur = self.agent_positions[agent].copy()  #current position
            vis = self.visibility(agent)
            if (vis,action) not in self.qtable.keys():   #if not in q table
                self.qtable[(vis,action)]=0
            #now choosing action according to exploration v exploitation
            action = -1
            randint = random.random()
            if randint<self.epsilon:
                action = np.random.choice(4)
            else:
                max = max(self.qtable[(vis,1)],self.qtable[(vis,2)],self.qtable[(vis,3)],self.qtable[(vis,4)])
                for i in range(1,5):
                    if self.qtable[tuple(vis,i)]==max:
                        action = i
                        break
                    
            reward = 0;
            # Move agent
            new_pos = cur.copy()
            new_vis = self.visibility(new_pos)
            # Move agent
            if action == 0:  # Up
                new_pos[1] = cur[1] - 1
            elif action == 1:  # Down
                new_pos[1] = cur[1] + 1
            elif action == 2:  # Left
                new_pos[0] = cur[0] - 1
            elif action == 3:  # Right
                new_pos[0] = cur[0] + 1
            # Check if the move is valid (within grid boundaries)
            if not(0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                reward = -2
            # Ensure no collisions with obstacles or other agents
            elif tuple(new_pos) in self.obstacles:
                reward = -2  # Penalize for hitting an obstacle
                print(f"{agent} hit an obstacle at {new_pos}, reward = {reward}")
            # Check if agent reached a target
            else:
                self.agent_positions[agent]=new_pos
                for target in self.target_positions:
                    if target in new_vis:
                        print(f"{agent} detected a target nearby at {target}!")
                        reward = 50
                        self.target_positions.remove(target)  # Remove the detected target
                        break  # Stop checking after finding one target

            for i in range(1,5):   # adding missing q vals if any
                if (new_vis,i) not in self.qtable.keys():
                    self.qtable[(new_vis,i)] = 0
            td = (reward + self.gamma * max(self.qtable[(new_vis,1)],self.qtable[(new_vis,2)],self.qtable[(new_vis,3)],self.qtable[(new_vis,4)]) - self.qtable[(vis,action)])

            self.qtable[(vis,action)] = self.qtable[(vis,action)] + td
        self.time = self.time + 1
    
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
        #adding obstacles
        visible.append(pos for pos in self.obstacles)
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
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False  # Press ESC to exit fullscreen

            self.update()
            self.render()

            # Stop when all targets are found
            if not self.target_positions:
                print("All targets found! Simulation complete.")
                break

            self.clock.tick(10)

        pygame.quit()

# Run the simulation
env = GridParallel()
#env.randomEnable = False #uncomment to srart exploiting
env.run()
