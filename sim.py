from copy import copy
from random import randint
from random import random
import matplotlib.pyplot as plt

class MazeSimulator:

    def __init__(self):
        
        self.maze = []

        self.num_row = 15
        self.num_col = 15

        self.agent_x = 1
        self.agent_y = 1

        '''
        Static goal location
        '''
        self.goal_x = 4
        self.goal_y = 5

        '''
        The VPG algorithm seems to be able to solve this
        '''
        # self.goal_x = 5 + randint(-1, 1)
        # self.goal_y = 5 + randint(-1, 1)

        '''
        But not this setting -- possibly because something like this would require memory? 
        '''
        # if random() > 0.5:
        #     self.goal_x = 5
        #     self.goal_y = 1
        # else:
        #     self.goal_x = 1
        #     self.goal_y = 5

        # generates an empty maze, W stands for a wall square
        top_bottom = ['W'] * self.num_col
        middle = ['W'] + [' '] * (self.num_col - 2) + ['W']
        self.maze.append(copy(top_bottom))
        for _ in range(self.num_row - 2):
            self.maze.append(copy(middle))
        self.maze.append(copy(top_bottom))
        self.maze[self.goal_y][self.goal_x] = 'G'
    
        # generates an information vector for each square
        self.maze_info = [[[] for c in range(self.num_col)] for r in range(self.num_row)]
        for y in range(self.num_col):
            for x in range(self.num_row):
                # find goal direction
                # currently encode left, right as -1, 1, and down, up as -1, 1
                goal_dir_encoding = [0, 0]
                if x < self.goal_x:
                    goal_dir_encoding[0] = 1
                elif x > self.goal_x:
                    goal_dir_encoding[0] = -1
                
                if y < self.goal_y:
                    goal_dir_encoding[1] = 1
                elif y > self.goal_y:
                    goal_dir_encoding[1] = -1

                # get distance to nearest wall in each direction [N, S, E, W]
                wall_dist = [0, 0, 0, 0]
                x_temp, y_temp = x, y
                while self.maze[y_temp][x] != 'W': # N
                    wall_dist[0] += 1
                    y_temp -= 1
                y_temp = y
                while self.maze[y_temp][x] != 'W': # S
                    wall_dist[1] += 1
                    y_temp += 1
                while self.maze[y][x_temp] != 'W': # E
                    wall_dist[2] += 1
                    x_temp += 1
                x_temp = x
                while self.maze[y][x_temp] != 'W': # W
                    wall_dist[3] += 1
                    x_temp -= 1
                self.maze_info[y][x] = [x, y]

    def reset_soft(self):
        '''
        keeps any environment instance-specific (randomly drawn) parameters the same, but resets the agent
        '''
        self.agent_x = 1
        self.agent_y = 1

    def __str__(self):
        s = ""
        for r in self.maze:
            s += "".join(r) + "\n"
        return s

    def step(self, action):
        '''
        action: 'N', 'S', 'E', or 'W' to move on the map
        return: next_state (vector, or None if terminal), reward (int)
        '''
        # move agent based on action
        delta = {'N': (0,-1),
                'S': (0,1),
                'E': (1,0),
                'W': (-1,0)}
        self.agent_x += delta[action][0]
        self.agent_y += delta[action][1]

        # revert action if unsuccessful
        if self.maze[self.agent_y][self.agent_x] == 'W':
            self.agent_x -= delta[action][0]
            self.agent_y -= delta[action][1]

        # calculate reward
        reward = 0
        if self.maze[self.agent_y][self.agent_x] == 'G':
            return None, 0
        else:
            # return self.get_state(), -((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)**(1/2)
            return self.get_state(), -1

    def get_state(self):
        '''
        returns the maze info vector corresponding to the agent's current x, y position
        '''
        return self.maze_info[self.agent_y][self.agent_x]

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


