import pygame
import numpy as np
import random

class GridParallel:
    def __init__(self):
        self.grid_size = 5
        self.num_obstacles = 4
        self.agent_positions = {"rescue1": np.array([0, 0]), "rescue2": np.array([4, 0])}
        self.target_positions = [tuple(np.array([4, 4])), tuple(np.array([0, 4]))]  # Targets as tuples
        self.obstacles = set()
        self._place_obstacles()

        self.done = {"rescue1": False, "rescue2": False}
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((300, 300))  # window size
        self.clock = pygame.time.Clock()

    def _place_obstacles(self):
        # Place obstacles ensuring they do not overlap with agents or targets
        while len(self.obstacles) < self.num_obstacles:
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            # Convert agent positions to tuples for proper comparison
            if pos not in [tuple(pos) for pos in self.agent_positions.values()] and pos not in self.target_positions:
                self.obstacles.add(pos)

    def render(self):
        self.screen.fill((255, 255, 255))  # fill screen with white

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 0), (obs[0]*60, obs[1]*60, 60, 60))

        # Draw agents
        for agent, pos in self.agent_positions.items():
            pygame.draw.circle(self.screen, (0, 0, 255), (pos[0]*60 + 30, pos[1]*60 + 30), 20)

        # Draw targets (only the remaining targets)
        for target in self.target_positions:
            pygame.draw.circle(self.screen, (255, 0, 0), (target[0]*60 + 30, target[1]*60 + 30), 20)

        pygame.display.flip()  # Update the display

    def update(self):
        actions = {agent: np.random.choice(4) for agent in self.agent_positions}

        for agent, action in actions.items():
            if self.done[agent]:  # Skip update if agent has reached a target
                continue

            new_pos = self.agent_positions[agent].copy()
            if action == 0:  # Up
                new_pos[1] = max(0, new_pos[1] - 1)
            elif action == 1:  # Down
                new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
            elif action == 2:  # Left
                new_pos[0] = max(0, new_pos[0] - 1)
            elif action == 3:  # Right
                new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)

            # Update position if not an obstacle
            if tuple(new_pos) not in self.obstacles:
                self.agent_positions[agent] = new_pos

            # Check if agent reached any target
            for target in self.target_positions[:]:  # Iterate over a copy of target_positions
                if tuple(self.agent_positions[agent]) == target:
                    self.done[agent] = True
                    self.target_positions.remove(target)  # Remove the found target
                    print(f"{agent} reached a target!")

        # Check if both agents are done
        if all(self.done.values()):
            print("Both agents have reached their target(s). Animation done.")

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Update the positions and check if both agents are done
            self.update()
            self.render()

            # Stop the animation when both agents have reached a target
            if all(self.done.values()):
                break

            self.clock.tick(10)  # 10 frames per second

        pygame.quit()

# Example usage
env = GridParallel()
env.run()
