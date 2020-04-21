from copy import copy
from random import randint
from random import random
from maze import Maze
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MazeSimulator:

    def __init__(self, num_rows = 2, num_cols = 2, num_walls = 0, start = (0,0), goal = (1,1)):
        self.maze = Maze(num_rows, num_cols, num_walls, start, goal)

        self.num_col = self.maze.num_cols
        self.num_row = self.maze.num_rows

        self.start = self.maze.start
        self.goal = self.maze.goal

        self.agent_row = self.start.row
        self.agent_col = self.start.col


        self.maze_info = self.generate_information_vector()


    def generate_information_vector(self):
        '''Generates an information vector for each square'''

        maze_info = [[[] for c in range(self.num_col)] for r in range(self.num_row)]

        for row in range(self.num_row):
            for col in range(self.num_col):
                # find goal direction
                # currently encode left, right as -1, 1, and down, up as -1, 1
                goal_dir_encoding = [0, 0]
                if row < self.goal.row:
                    goal_dir_encoding[0] = 1
                elif row > self.goal.row:
                    goal_dir_encoding[0] = -1
                
                if col < self.goal.col:
                    goal_dir_encoding[1] = 1
                elif col > self.goal.col:
                    goal_dir_encoding[1] = -1

                # get distance to nearest wall in each direction [N, S, E, W]
                wall_dist = [0, 0, 0, 0]
                row_temp, col_temp = row, col
                while row_temp >= 0 and not self.maze.get_cell(row_temp, col).is_wall(): # N
                    wall_dist[0] += 1
                    row_temp -= 1
                row_temp = row
                while row_temp < self.num_row and not self.maze.get_cell(row_temp, col).is_wall(): # S
                    wall_dist[1] += 1
                    row_temp += 1
                while not col_temp < self.num_col and self.maze.get_cell(row, col_temp).is_wall(): # E
                    wall_dist[2] += 1
                    col_temp += 1
                col_temp = col
                while col_temp > 0 and not self.maze.get_cell(row, col_temp).is_wall(): # W
                    wall_dist[3] += 1
                    col_temp -= 1
                maze_info[row][col] = [row, col]

        return maze_info

    def reset_soft(self):
        '''
        keeps any environment instance-specific (randomly drawn) parameters the same, but resets the agent
        '''
        self.agent_row = self.maze.start.row
        self.agent_col = self.maze.start.col

    def __str__(self):
        s = ""
        count = 0
        for row in range(self.num_row):
            r = ""
            for col in range(self.num_col):
                if row == self.agent_row and col == self.agent_col:
                    r += "*"
                else:
                    r += str(self.maze.maze[row][col])
            r += "\n"
            s += r
        return s

    def step(self, action):
        '''
        action: 'N', 'S', 'E', or 'W' to move on the map
        return: next_state (vector, or None if terminal), reward (int)
        '''
        # move agent based on action
        print ("old: ", self.agent_row, self.agent_col)
        print ("action: ", action)
        delta = {'N': (-1,0),
                'S': (1,0),
                'E': (0,1),
                'W': (0,-1)}
        self.agent_row += delta[action][0]
        self.agent_col += delta[action][1]

        # if self.agent_row < 0 or self.agent_row >= self.num_row or self.agent_col < 0 or self.agent_col >= self.num_col:
        #     self.agent_row -= delta[action][0]
        #     self.agent_col -= delta[action][1]
        #     return self.get_state(), -((self.agent_row - self.goal.row)**2 + (self.agent_col - self.goal.col)**2)**(1/2)


        # revert action if unsuccessful
        if self.maze.get_cell(self.agent_row, self.agent_col).is_wall():
            # print ("yes: ", self.agent_row, self.agent_col)
            self.agent_row -= delta[action][0]
            self.agent_col -= delta[action][1]
            return self.get_state(), -((self.agent_row - self.goal.row)**2 + (self.agent_col - self.goal.col)**2)**(1/2)
        else:
            self.agent_row += delta[action][0]
            self.agent_col += delta[action][1]

        print ("new: ", self.agent_row, self.agent_col)

        # calculate reward
        reward = 0
        if self.maze.get_cell(self.agent_row, self.agent_col).is_goal():
            return None, 10
        else:
            return self.get_state(), -((self.agent_row - self.goal.row)**2 + (self.agent_col - self.goal.col)**2)**(1/2)
            # return self.get_state(), -1

    def get_state(self):
        '''
        returns the maze info vector corresponding to the agent's current x, y position
        '''
        #return self.maze_info[self.agent_row][self.agent_col]
        loc = np.reshape([self.agent_row, self.agent_col], (1, 2))
        return [self.agent_row, self.agent_col]

    def visualize(self, policy, i):
        '''
        Visualize a policy's decisions in a heatmap fashion
        '''

        # lets make a (row*3)x(col*3) heatmap for the policies decisions


        heatmap = [[0 for c in range(self.num_col)] for r in range(self.num_row)]
        action_space = {0: "N", 1: "S", 2: "E", 3: "W"}
        offsets = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)} # x, y offsets for heatmap
        for row in range(1, self.num_row, 2):
            for col in range(1, self.num_col, 2):
                print (row, col)
                if self.maze.maze[row][col].is_empty_cell():
                    heatmap[row][col] = 1

                # get action probs at this state
                action_probs = policy(torch.as_tensor(self.get_state(), dtype=torch.float32))
                for a in [0, 1, 2, 3]: # action space
                    row_loc = row+offsets[a][0]
                    col_loc = col+offsets[a][1]
                    print ("a: ", a, action_probs[a].item())
                    heatmap[row_loc][col_loc] = action_probs[a].item()
        
        print (heatmap)
        plt.imshow(np.array(heatmap), cmap='Blues', interpolation='nearest')
        plt.savefig("Iteration%sHeatmap_(%s, %s)" % (i, self.agent_row, self.agent_col))
        plt.clf()



# class MazeSimulator_Default:

#     def __init__(self):
        
#         self.maze = []

#         self.num_row = 7
#         self.num_col = 7

#         '''
#         Static goal location
#         '''
#         self.goal_x = 3
#         self.goal_y = 3

#         self.agent_x = randint(1, 5)
#         self.agent_y = randint(1, 5)
#         if abs(self.agent_x - self.goal_x) < 2 and abs(self.agent_y - self.goal_y) < 2:
#             self.agent_x = 1
#             self.agent_y = 1 # move off goal state
#         self.initial_x = self.agent_x
#         self.initial_y = self.agent_y

#         '''
#         The VPG algorithm seems to be able to solve this
#         '''
#         # self.goal_x = 5 + randint(-1, 1)
#         # self.goal_y = 5 + randint(-1, 1)

#         '''
#         But not this setting -- possibly because something like this would require memory? 
#         '''
#         # if random() > 0.5:
#         #     self.goal_x = 5
#         #     self.goal_y = 1
#         # else:
#         #     self.goal_x = 1
#         #     self.goal_y = 5

#         # generates an empty maze, W stands for a wall square
#         top_bottom = ['W'] * self.num_col
#         middle = ['W'] + [' '] * (self.num_col - 2) + ['W']
#         self.maze.append(copy(top_bottom))
#         for _ in range(self.num_row - 2):
#             self.maze.append(copy(middle))
#         self.maze.append(copy(top_bottom))
#         self.maze[self.goal_y][self.goal_x] = 'G'
    
#         # generates an information vector for each square
#         self.maze_info = [[[] for c in range(self.num_col)] for r in range(self.num_row)]
#         for y in range(self.num_col):
#             for x in range(self.num_row):
#                 # find goal direction
#                 # currently encode left, right as -1, 1, and down, up as -1, 1
#                 goal_dir_encoding = [0, 0]
#                 if x < self.goal_x:
#                     goal_dir_encoding[0] = 1
#                 elif x > self.goal_x:
#                     goal_dir_encoding[0] = -1
                
#                 if y < self.goal_y:
#                     goal_dir_encoding[1] = 1
#                 elif y > self.goal_y:
#                     goal_dir_encoding[1] = -1

#                 # get distance to nearest wall in each direction [N, S, E, W]
#                 wall_dist = [0, 0, 0, 0]
#                 x_temp, y_temp = x, y
#                 while self.maze[y_temp][x] != 'W': # N
#                     wall_dist[0] += 1
#                     y_temp -= 1
#                 y_temp = y
#                 while self.maze[y_temp][x] != 'W': # S
#                     wall_dist[1] += 1
#                     y_temp += 1
#                 while self.maze[y][x_temp] != 'W': # E
#                     wall_dist[2] += 1
#                     x_temp += 1
#                 x_temp = x
#                 while self.maze[y][x_temp] != 'W': # W
#                     wall_dist[3] += 1
#                     x_temp -= 1
#                 self.maze_info[y][x] = [x, y]

#     def reset_soft(self):
#         '''
#         keeps any environment instance-specific (randomly drawn) parameters the same, but resets the agent
#         '''
#         self.agent_x = self.initial_x
#         self.agent_y = self.initial_y

#     def __str__(self):
#         s = ""
#         count = 0
#         for r in self.maze:
#             if count == self.agent_y:
#                 s += "".join(r[0:self.agent_x])
#                 s += "*"
#                 s += "".join(r[self.agent_x+1:self.num_col]) + "\n"
#             else:
#                 s += "".join(r) + "\n"
#             count += 1
#         return s

#     def step(self, action):
#         '''
#         action: 'N', 'S', 'E', or 'W' to move on the map
#         return: next_state (vector, or None if terminal), reward (int)
#         '''
#         # move agent based on action
#         delta = {'N': (0,-1),
#                 'S': (0,1),
#                 'E': (1,0),
#                 'W': (-1,0)}
#         self.agent_x += delta[action][0]
#         self.agent_y += delta[action][1]

#         # revert action if unsuccessful
#         if self.maze[self.agent_y][self.agent_x] == 'W':
#             self.agent_x -= delta[action][0]
#             self.agent_y -= delta[action][1]

#         # calculate reward
#         reward = 0
#         if self.maze[self.agent_y][self.agent_x] == 'G':
#             return None, 0
#         else:
#             return self.get_state(), -((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)**(1/2)
#             # return self.get_state(), -1

#     def get_state(self):
#         '''
#         returns the maze info vector corresponding to the agent's current x, y position
#         '''
#         return self.maze_info[self.agent_y][self.agent_x]

#     def visualize(self, policy, i):
#         '''
#         Visualize a policy's decisions in a heatmap fashion
#         '''

#         # lets make a (row*3)x(col*3) heatmap for the policies decisions

#         heatmap = [[0 for c in range(3*self.num_col)] for r in range(3*self.num_row)]
#         action_space = {0: "N", 1: "S", 2: "E", 3: "W"}
#         offsets = {0: (1, 0), 1: (1, 2), 2: (2, 1), 3: (0, 1)} # x, y offsets for heatmap
#         for y in range(1, self.num_row-1):
#             for x in range(1, self.num_col-1):
#                 heatmap[3*y + 1][3*x + 1] = 0.5

#                 upper_left = (x * 3, y * 3) # in x, y

#                 # get action probs at this state
#                 action_probs = policy(torch.as_tensor(self.maze_info[y][x], dtype=torch.float32))
#                 for a in [0, 1, 2, 3]: # action space
#                     x_loc = upper_left[0] + offsets[a][0]
#                     y_loc = upper_left[1] + offsets[a][1]
#                     heatmap[y_loc][x_loc] = action_probs[a].item()
        
#         plt.imshow(np.array(heatmap), cmap='Blues', interpolation='nearest')
#         plt.savefig("Iteration%sHeatmap" % (i))
#         plt.clf()

        


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
        x = np.array([self.agent_x])
        # x = np.reshape(x, (1,1))
        return x

    def visualize(self, policy, i):
        '''
        Visualize a policy's decisions in a heatmap fashion
        '''

        # lets make a (row*3)x(col*3) heatmap for the policies decisions


        heatmap = [0 for i in range(24)]
        action_space = {0: "R", 1: "L"}
        offsets = {0: 1, 1: -1} # x, y offsets for heatmap
        for row in range(1, 24, 3):

                heatmap[row] = 1
                # get action probs at this state
                action_probs = policy(torch.as_tensor(self.get_state(), dtype=torch.float32))
                for a in [0, 1]: # action space
                    x_loc = row+offsets[a]
                    heatmap[x_loc] = action_probs[a].item()
        
        plt.imshow(np.reshape(np.array(heatmap), (1,24)), cmap='Blues', interpolation='nearest')
        plt.savefig("ShortCorridor_Iteration%sHeatmap_%s)" % (i, self.agent_x))
        plt.clf()

# if __name__ == "__main__":


#     class Policy(nn.Module):

#         def __init__(self):
#             super(Policy, self).__init__()

#             self.input_size = 2
#             self.hidden_size = 100

#             self.ACTION_SPACE = {0: "N", 1: "S", 2: "E", 3: "W"}

#             self.num_actions = len(self.ACTION_SPACE.keys())

#             self.fc1 = nn.Linear(self.input_size, self.hidden_size)
#             self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
#             # self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
#             # self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)

#             self.fc5 = nn.Linear(self.hidden_size, self.num_actions)
#             self.softmax = nn.Softmax()


#         def forward(self, x):
#             '''
#             x: input vector describing state
#             return: vector containing probabilities?? of each
#             '''
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             # x = F.relu(self.fc3(x))
#             # x = F.relu(self.fc4(x))

#             x = self.fc5(x)
#             return self.softmax(x)

#     world = MazeSimulator(2, 2, 0, (0,0), (1,1))
#     policy = Policy()
#     world.visualize(policy, 0)
    # simple test case
    # world = MazeSimulator(7, 7, 0, start = (0,0), goal = (3,3))
    # world.step('N')
    # print(world.agent_row, world.agent_col) # the agent should still be at (1, 1) since it hit a wall
    # world.step('S')
    # world.step('S')
    # world.step('S')
    # world.step('S')
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


    # mean_rewards = []
    # upper_range = 100
    # for i in range(1, upper_range):
    #     epsilon = i/upper_range

    #     num_trials = 600
    #     all_rewards = []
    #     for n in range(num_trials):
    #         state = 0
    #         total_reward = 0
    #         num_steps = 0
    #         env = ShortCorridor()
    #         while state != None and num_steps < 50:
    #             num_steps += 1
    #             if random() < epsilon:
    #                 state, reward = env.step('R')
    #             else:
    #                 state, reward = env.step('L')
    #             total_reward += reward
    #         all_rewards.append(total_reward)
    #     mean_reward = sum(all_rewards)/len(all_rewards)
    #     mean_rewards.append(mean_reward)

    # plt.plot(list(range(1, upper_range)), mean_rewards)
    # plt.show()


