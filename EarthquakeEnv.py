import numpy as np
import random
import matplotlib.pyplot as plt
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete
from gym import spaces

class EarthquakeEnv(ParallelEnv):
    dimensions = 8
    obstacles_count = 3
    survivor_count = 2
    def __init__(self, num_agents=2 ,Dimensions=dimensions, obstacles=obstacles_count, survivors=survivor_count, timeLimit = 50):
        super().__init__() #inheriting parallelEnv class
        #defining the environment description 
        self.dimensions=Dimensions
        self.obstacles_count = obstacles
        self.survivor_count = survivors
        self.timeLimit = timeLimit
        self.time = 0
        
        #obstacles
        self.obstacles=["obstacle"+str(num) for num in range(1,self.obstacles_count+1)] #obstacle names- list
        self.obsPos = dict()
        for obst in self.obstacles:      #position of obstacles- dict--- Obstaclename: (x,y)  
            self.obsPos[obst] = (random.randint(0, self.dimensions-1), random.randint(0, self.dimensions-1))
        
        #survivors
        self.survivors=["survivor"+str(num) for num in range(1,self.survivor_count+1)] #survivor names- list
        self.survPos = dict()           #position of survivors- dict--- Survivorname: (x,y)  
        for surv in self.survivors:      #to prevent survivor from spawning on obstacle
            while True:
                spawn = (random.randint(0, self.dimensions-1), random.randint(0, self.dimensions-1))
                if spawn in self.obsPos.values():
                    continue
                else:
                    self.survPos[surv] = spawn
                    break
        
        #agents        
        self.agents=["agent"+str(num) for num in range(1,num_agents+1)] #agent names list
        self.agentPos = dict()                      #position of agents- dict--- Agentname: (x,y)
        for ag in self.agents:              #to prevent agents from spawning on obstacles or survivors
            while True:
                spawn = (random.randint(0, self.dimensions-1), random.randint(0, self.dimensions-1))
                if spawn in self.obsPos.values() or spawn in self.survPos.values():
                    continue
                else:
                    self.agentPos[ag]=spawn
                    break

        self.action_spaces = {ag:Discrete(4) for ag in self.agents}
        self.observation_spaces = {}
        
        

    def display(self):
        pass
    
    def step(self, actions):
        return super().step(actions)
    
    def render(self):
        return super().render()
    
    def reset(self, seed = None, options = None):
        return super().reset(seed, options)
    
    
    