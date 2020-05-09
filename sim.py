from copy import copy
from random import randint
from random import random
import matplotlib.pyplot as plt
import torch
import numpy as np

class MazeSimulator:

    def __init__(self, goal_X, goal_Y, reward_type, state_rep, maze = None, wall_penalty=0, normalize_state=True):
        
        self.maze = []

        self.num_row = 16
        self.num_col = 9

        self.agent_x = 1
        self.agent_y = 1

        self.mean_x = self.num_col / 2
        self.std_dev_x = np.sqrt(1/12*(self.num_col)**2) # assuming uniform distribution over possible vals
        self.mean_y = self.num_row / 2
        self.std_dev_y = np.sqrt(1/12*(self.num_row)**2)

        self.reward = reward_type
        self.wall_penalty = wall_penalty
        self.normalize_state = normalize_state

        self.state_rep = state_rep
        self.state_rep_func = {"onehot": self.__get_state_onehot_xy,
                        "fullboard": self.__get_state_fullboard_xy,
                        "xy": self.__get_state_xy}[self.state_rep]

        self.initial_x = self.agent_x
        self.initial_y = self.agent_y

        self.action_space = {0: "N", 1: "S", 2: "E", 3: "W"}

        self.num_actions = len(self.action_space)
        if self.state_rep == "fullboard":
            self.state_size = self.num_row * self.num_col
        elif self.state_rep == "onehot":
            self.state_size = self.num_row + self.num_col
        elif self.state_rep == "xy":
            self.state_size = 2
        
        self.state_size += 4
        
        if maze == None:
            self.maze = [["W", "W", "W", "W", "W", "W", "W", "W", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", " ", " ", " ", " ", " ", " ", " ", "W"],
                        ["W", "W", "W", "W", "W", "W", "W", "W", "W"]]
        else:
            self.maze = maze

        self.goal_x = goal_X
        self.goal_y = goal_Y

        self.maze[self.goal_y][self.goal_x] = 'G'

        # generates an information vector for each square
        self.maze_info = [[[] for c in range(self.num_col)] for r in range(self.num_row)]
        # print(self.maze_info)
        for x in range(1, self.num_col-1):
            for y in range(1, self.num_row-1):
                '''
                We don't actually do anything with this wall information yet
                '''
                # check if a wall is in each direction
                walls = [0, 0, 0, 0]
                if self.maze[y + 1][x] == "W":
                    walls[0] = 1
                if self.maze[y - 1][x] == "W":
                    walls[1] = 1
                if self.maze[y][x + 1] == "W":
                    walls[2] = 1
                if self.maze[y][x - 1] == "W":
                    walls[3] = 1
                
                self.maze_info[y][x] = self.state_rep_func(x, y) + walls

    def __get_action(self, policy_output):
        return self.action_space[policy_output.item()]

    def generate_fresh(self):
        # self.reset_soft()
        return MazeSimulator(self.goal_x, self.goal_y, self.reward, self.state_rep, self.maze, self.wall_penalty, self.normalize_state)

    def reset_soft(self):
        '''
        keeps any environment instance-specific (randomly drawn) parameters the same, but resets the agent
        '''
        self.agent_x = self.initial_x
        self.agent_y = self.initial_y

    def __str__(self):
        s = ""
        for r in self.maze:
            s += "".join(r) + "\n"
        return s

    def step(self, policy_output):
        '''
        action: 'N', 'S', 'E', or 'W' to move on the map
        return: next_state (vector, or None if terminal), reward (int)
        '''
        action = self.__get_action(policy_output)

        # move agent based on action
        delta = {'N': (0,-1),
                'S': (0,1),
                'E': (1,0),
                'W': (-1,0)}
        self.agent_x += delta[action][0]
        self.agent_y += delta[action][1]

        penalty = 0
        # revert action if unsuccessful
        if self.maze[self.agent_y][self.agent_x] == 'W':
            self.agent_x -= delta[action][0]
            self.agent_y -= delta[action][1]
            penalty = self.wall_penalty

        if self.maze[self.agent_y][self.agent_x] == 'G':
            return self.get_state(), 0
        else:
            if self.reward == "distance":
                return self.get_state(), penalty-((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)**(1/2)
            elif self.reward == "constant":
                return self.get_state(), penalty-1

    def get_state(self):
        '''
        returns the maze info vector corresponding to the agent's current x, y position
        '''
        if self.maze[self.agent_y][self.agent_x] == 'G':
            return None
        else:
            return self.maze_info[self.agent_y][self.agent_x]

    def __get_state_xy(self, x, y):
        '''
        returns the maze info vector corresponding to the agent's current x, y position
        '''
        if self.normalize_state:
            return [(x - self.mean_x) / self.std_dev_x, (y - self.mean_y) / self.std_dev_y]
        else:
            return [x, y]

    def __get_state_onehot_xy(self, x, y):
        l = [0] * (self.num_row + self.num_col)
        l[x] = 1
        l[y + self.num_col] = 1
        return l

    def __get_state_fullboard_xy(self, x, y):
        l = [0] * (self.num_row * self.num_col)
        l[y*self.num_col + x] = 1
        # l[y + self.num_row] = 1
        return l

    def visualize(self, policy, title):
        '''
        Visualize a policy's decisions in a heatmap fashion
        '''

        heatmap = [[0 for c in range(3*self.num_col)] for r in range(3*self.num_row)]
        offsets = {0: (1, 0), 1: (1, 2), 2: (2, 1), 3: (0, 1)} # x, y offsets for heatmap
        for y in range(1, self.num_row-1):
            for x in range(1, self.num_col-1):
                if self.maze[y][x] != "W":                    
                    heatmap[3*y + 1][3*x + 1] = 0.5

                    upper_left = (x * 3, y * 3) # in x, y

                    # get action probs at this state
                    action_probs = policy(torch.as_tensor(self.maze_info[y][x], dtype=torch.float32))
                    for a in [0, 1, 2, 3]: # action space
                        x_loc = upper_left[0] + offsets[a][0]
                        y_loc = upper_left[1] + offsets[a][1]
                        heatmap[y_loc][x_loc] = action_probs[a].item()


        plt.imshow(np.array(heatmap), cmap='PRGn', interpolation='nearest')
        plt.savefig(title)
        plt.clf()

    def visualize_value(self, critic, title):
        '''
        Visualize the value of each state
        '''

        heatmap = [[0 for c in range(self.num_col)] for r in range(self.num_row)]
        for y in range(1, self.num_row-1):
            for x in range(1, self.num_col-1):
                if self.maze[y][x] == "W":
                    heatmap[y][x] = 0
                else:
                    heatmap[y][x] = critic(torch.as_tensor(self.maze_info[y][x], dtype=torch.float32)).item()

        plt.imshow(np.array(heatmap), cmap='PRGn', interpolation='nearest')
        plt.savefig(title)
        plt.clf()

class MazeArgs():

    def __init__(self):
        self.rows = None
        self.cols = None
        self.goal = None
        self.agent = None

class Discrete2D:
    state_size = 2
    num_actions = 4

    def __init__(self, args):
        self.args = args

        self.rows = args.rows
        self.cols = args.cols
        self.dims = np.array([self.cols, self.rows])

        self.agent = np.array(args.agent) # list
        self.goal = np.array(args.goal) # list

    def get_state(self):
        '''
        ret: list [x, y]
        '''
        return list(self.agent/self.dims)

    def step(self, policy_output):
        '''
        input: int
        ret: state (list), reward (int)
        '''
        x = self.agent[0]
        y = self.agent[1]
        if policy_output == 0:
            self.agent[0] = min(x+1, self.cols)
        if policy_output == 1:
            self.agent[0] = max(x-1, 0)

        if policy_output == 2:
            self.agent[1] = min(y+1, self.rows)
        if policy_output == 3:
            self.agent[1] = max(y-2, 0)

        dist_to_goal = -np.sqrt(np.sum((self.agent - self.goal)**2))
        if dist_to_goal == 0:
            return None, 0
        else:
            return self.get_state(), dist_to_goal
    
    def generate_fresh(self):
        return Discrete2D(self.args)



class Continuous2D:
    state_size = 2
    num_actions = 2

    def __init__(self, args):
        self.args = args

        # self.dims = np.array([self.cols, self.rows])

        self.agent = np.array(args.agent) # list
        self.goal = np.array(args.goal) # list

    def get_state(self):
        '''
        ret: list [x, y]
        '''
        return list(self.agent)

    def step(self, policy_output):
        '''
        input: int
        ret: state (list), reward (int)
        '''
        actions = np.array(torch.clamp(policy_output, -0.1, 0.1))
        self.agent += actions

        dist_to_goal = -np.sqrt(np.sum((self.agent - self.goal)**2))
        if abs(dist_to_goal) <= 0.01:
            return None, 0
        else:
            return self.get_state(), dist_to_goal
    
    def generate_fresh(self):
        return Continuous2D(self.args)






class Discrete2DMazeFlags:

    def __init__(self, args):
        self.args = args

        self.rows = args.rows
        self.cols = args.cols
        self.dims = np.array([self.cols, self.rows])

        self.agent = np.array(args.agent) # list
        self.goal = np.array(args.goal) # list

        self.state_size = 4
        self.num_actions = 4

    def get_state(self):
        '''
        ret: list [x, y]
        '''
        direction = (np.zeros(self.agent.shape) + (self.agent > self.goal))
        return list(self.agent/self.dims) + list(direction)

    def step(self, policy_output):
        '''
        input: int
        ret: state (list), reward (int)
        '''
        x = self.agent[0]
        y = self.agent[1]
        if policy_output == 0:
            self.agent[0] = min(x+1, self.cols)
        if policy_output == 1:
            self.agent[0] = max(x-1, 0)

        if policy_output == 2:
            self.agent[1] = min(y+1, self.rows)
        if policy_output == 3:
            self.agent[1] = max(y-2, 0)

        dist_to_goal = -np.sqrt(np.sum((self.agent - self.goal)**2))
        if dist_to_goal == 0:
            return None, 0
        else:
            return self.get_state(), dist_to_goal
    
    def generate_fresh(self):
        return Discrete2DMazeFlags(self.args)
