from copy import copy
from random import randint
from random import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.autograd import Variable


class MazeSimulator:

    def __init__(self, goal_X, goal_Y, reward_type, state_rep, maze = None, wall_penalty=0, normalize_state=True):
        
        self.maze = []

        self.num_row = 16
        self.num_col = 9

        # self.hx = torch.zeros(1, 300)
        # self.cx = torch.zeros(1, 300)

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

        self.num_actions = 4
        if self.state_rep == "fullboard":
            self.state_size = self.num_row * self.num_col
        elif self.state_rep == "onehot":
            self.state_size = self.num_row + self.num_col
        elif self.state_rep == "xy":
            self.state_size = 2
        
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
                
                self.maze_info[y][x] = self.state_rep_func(x, y)

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
            # print ("GOAL")
            return self.state_rep_func(self.agent_x, self.agent_y), 100 #self.get_state(), 100
        else:
            # print ("NOT GOAL")
            if self.reward == "distance":
                # print ("state: ", self.get_state())
                return self.state_rep_func(self.agent_x, self.agent_y), penalty-((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)**(1/2) #self.get_state(), penalty-((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)**(1/2)
            elif self.reward == "constant":
                return self.state_rep_func(self.agent_x, self.agent_y), penalty-1

    def get_state(self):
        '''
        returns the maze info vector corresponding to the agent's current x, y position
        '''
        # print ("maze_info: ", self.maze_info)
        if self.maze[self.agent_y][self.agent_x] == 'G':
            return self.maze_info[self.agent_y][self.agent_x]
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

    def visualize(self, policy, savefile="Heatmap"):
        '''
        Visualize a policy's decisions in a heatmap fashion
        '''

        heatmap = [[0 for c in range(3*self.num_col)] for r in range(3*self.num_row)]
        # print ("col: ", self.num_col)
        # print ("row: ", self.num_row)
        offsets = {0: (1, 0), 1: (1, 2), 2: (2, 1), 3: (0, 1)} # x, y offsets for heatmap
        for y in range(1, self.num_row-1):
            for x in range(1, self.num_col-1):
                if self.maze[y][x] != "W":                    
                    heatmap[3*y + 1][3*x + 1] = 0.5

                    upper_left = (x * 3, y * 3) # in x, y

                    # get action probs at this state
                    tensor = torch.from_numpy(np.array(self.state_rep_func(x, y))).float()
                    # print ("tensor: ", tensor)
                    _, action_probs, _ = policy((Variable(tensor), (policy.hx, policy.cx)))
                    print ("action_probs: ", action_probs[0].detach().numpy())
                    for a in [0, 1, 2, 3]: # action space
                        x_loc = upper_left[0] + offsets[a][0]
                        y_loc = upper_left[1] + offsets[a][1]
                        # print ((y_loc, x_loc, 0), action_probs[0].detach().numpy()[a])
                        heatmap[y_loc][x_loc] = action_probs[0].detach().numpy()[a]


        plt.imshow(np.array(heatmap), cmap='Blues', interpolation='nearest')
        plt.savefig(savefile)
        plt.clf()

    def visualize_value(self, critic, savefile="Valuemap"):
        '''
        Visualize the value of each state
        '''

        heatmap = [[0 for c in range(self.num_col)] for r in range(self.num_row)]
        for y in range(1, self.num_row-1):
            for x in range(1, self.num_col-1):
                if self.maze[y][x] == "W":
                    heatmap[y][x] = 0
                else:
                    tensor = torch.from_numpy(np.array(self.state_rep_func(x, y))).float()
                    # print ("tensor: ", tensor)
                    value, _, _ = critic((Variable(tensor), (critic.hx, critic.cx)))#[1].detach().numpy()
                    print ("value ", (y, x), value[0].detach().numpy()[0])
                    heatmap[y][x] = value[0].detach().numpy()[0] #.item()

        plt.imshow(np.array(heatmap), cmap='Blues', interpolation='nearest')
        plt.savefig(savefile)
        plt.clf()






class ShortCorridor:

    def __init__(self):

        self.agent_x = 0

        self.goal_x = 6

        self.reverse_states = [1, 3]

    def step(self, action):
        '''
        action: 'N', 'S', 'E', or 'W' to move on the map
        return: next_state (vector, or None if terminal), reward (int)
        '''
        if action == 'L':
            if self.agent_x == 0:
                self.agent_x == self.goal_x

            if self.agent_x in self.reverse_states:
                self.agent_x += 1 # left goes right
            else:
                self.agent_x -= 1
        elif action == 'R':
            if self.agent_x in self.reverse_states:
                self.agent_x -= 1
            else:
                self.agent_x += 1
        
        if self.agent_x < 0:
            self.agent_x = 0
        if self.agent_x > self.goal_x:
            self.agent_x = self.goal_x

        if self.agent_x == self.goal_x:
            return None, 0
        else:
            return self.get_state(), self.agent_x - self.goal_x

    def get_state(self):
        '''
        returns the maze info vector corresponding to the agent's current x, y position
        '''
        # l = [0, 0, 0, 0, 0]
        # l[self.agent_x] = 1
        # return l
        return [self.agent_x]

if __name__ == "__main__":
    # # simple test case
    # world = WorldSimulator()
    # world.step('N')
    # print(world.agent_x, world.agent_y) # the agent should still be at (1, 1) since it hit a wall
    # world.step('S')
    # world.step('S')
    # print(world.get_state()) # in the goal row, so the y goal direction should be zero
    # for i in range(0, 6):
    #     print(world.step('E')) # last reward should be 20
    # print(world.get_state()) # at goal location, so the first two indices should both be zero



    # print("*****")
    # env = ShortCorridor()
    # print(env.step("L"), env.agent_x) # -1 (0)
    # print(env.step("R"), env.agent_x) # -1 (1)
    # print(env.step("R"), env.agent_x) # -1 (0)
    # print(env.step("R"), env.agent_x) # -1 (1)
    # print(env.step("L"), env.agent_x) # -1 (2)
    # print(env.step("R"), env.agent_x) # None (3)


    mean_rewards = []
    upper_range = 100
    for i in range(1, upper_range):
        epsilon = i/upper_range

        num_trials = 600
        all_rewards = []
        for n in range(num_trials):
            state = 0
            total_reward = 0
            num_steps = 0
            env = ShortCorridor()
            while state != None and num_steps < 50:
                num_steps += 1
                if random() < epsilon:
                    state, reward = env.step('R')
                else:
                    state, reward = env.step('L')
                total_reward += reward
            all_rewards.append(total_reward)
        mean_reward = sum(all_rewards)/len(all_rewards)
        mean_rewards.append(mean_reward)

    plt.plot(list(range(1, upper_range)), mean_rewards)
    plt.show()


'''
    observation is the current 2D position

    reward is negative distance to goal

    actions are velocity commands clipped to [-0.1, 0.1]

    Horizon is 100, environment ended when agent was within 0.01 of goal
'''