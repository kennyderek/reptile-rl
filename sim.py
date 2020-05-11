import copy
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
    state_size = 5*7
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
        return 

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


class SideScroller:
    state_size = 6*6
    num_actions = 4

    def __init__(self, args):
        self.args = args
        self.rows = args.rows
        self.cols = args.cols

        self.blockers = args.blockers # set of (x, y) arrays?
        
        self.agent = np.array([0, self.rows-1]) # bottom left location
        self.agent_velocity = np.array([0, 0])

        self.screen = [[0 for i in range(self.cols)] for i in range(self.rows)]

        # populate the screen image, with a 1 for the agent, -1 for blockers, 0 otherwise
        self.screen[self.agent[1]][self.agent[0]] = 1
        for block in self.blockers:
            self.screen[block[1]][block[0]] = -1
        # self.agent_screen = [[0 for i in range(self.cols)] for i in range(self.rows)]
        self.goal = np.array([self.cols - 1, self.rows - 1])
        self.prev_screen = copy.deepcopy(self.screen)

    def get_state(self):
        '''
        ret: image?
        '''
        l = []
        for r, rp in zip(self.screen, self.prev_screen):
            l += list(np.array(r) + 0.5*np.array(rp))
        return l
        # return [[self.screen]]
        # return [[list(np.array(self.screen) + 0.5 * np.array(self.prev_screen))]]

    def step(self, policy_output):
        '''
        input: int
        ret: state (list), reward (int)
        '''
        self.prev_screen = copy.deepcopy(self.screen)
        policy_output = policy_output.item()
        x = self.agent[0]
        y = self.agent[1]
        if policy_output == 0 and self.agent[1] == self.rows - 1: # going up with a jump, and not in air currently
            self.agent_velocity[1] = -2

        if policy_output == 1: # right
            self.agent_velocity[0] = min(self.agent_velocity[0] + 1, 2)

        if policy_output == 2: # left
            self.agent_velocity[0] = max(self.agent_velocity[0] - 1, -2)

        if policy_output == 3: # down, does not do anything
            pass
        
        # add velocity to the agent position
        self.agent_old = copy.copy(self.agent)
        vel_x = self.agent_velocity[0]
        vel_y = self.agent_velocity[1]
        for i in range(abs(vel_x)):
            if [self.agent[0] + np.sign(vel_x), self.agent[1]] not in self.blockers:
                self.agent[0] += np.sign(vel_x)
            else:
                self.agent_velocity[0] = 0
                break
        
        for i in range(abs(vel_y)):
            if [self.agent[0], self.agent[1] + np.sign(vel_y)] not in self.blockers:
                self.agent[1] += np.sign(vel_y)
            else:
                self.agent_velocity[1] = 0
                break
        

        self.agent[0] = max(min(self.agent[0], self.cols-1), 0)
        self.agent[1] = max(min(self.agent[1], self.rows-1), 0)

        self.screen[self.agent_old[1]][self.agent_old[0]] = 0
        self.screen[self.agent[1]][self.agent[0]] = 1

        # simulate forces of gravity
        if self.agent[1] < self.rows - 1:
            self.agent_velocity[1] = self.agent_velocity[1] + 1
        else:
            self.agent_velocity[1] = 0

        if list(self.agent) == list(self.goal):
            return None, 0
        else:
            return self.get_state(), -1
    
    def generate_fresh(self):
        return SideScroller(self.args)

    def plot(self):
        print(np.array(self.screen))


class Gobble:
    state_size = 6*6
    num_actions = 4

    def __init__(self, args):
        self.args = args
        self.rows = args.rows
        self.cols = args.cols

        # self.blockers = args.blockers # set of (x, y) arrays?
        self.targets = set([tuple(t) for t in args.targets])
        
        self.agent = np.array([0, self.rows-1]) # bottom left location
        # self.agent_velocity = np.array([0, 0])

        self.screen = [[0 for i in range(self.cols)] for i in range(self.rows)]

        # populate the screen image, with a 1 for the agent, -1 for blockers, 0 otherwise
        self.screen[self.agent[1]][self.agent[0]] = 1
        for target in self.targets:
            self.screen[target[1]][target[0]] = -1
        
        self.prev_screen = copy.deepcopy(self.screen)
        # self.goal = np.array([self.cols - 1, self.rows - 1])

    def get_state(self):
        '''
        ret: image?
        '''
        l = []
        for r, rp in zip(self.screen, self.prev_screen):
            l += list(np.array(r) + 0.5*np.array(rp))
        return l
        # return [[list(np.array(self.screen) + 0.5 * np.array(self.prev_screen))]]

    def step(self, policy_output):
        '''
        input: int
        ret: state (list), reward (int)
        '''
        reward_mod = 0
        policy_output = policy_output.item()
        self.prev_screen = copy.deepcopy(self.screen)
        # x = self.agent[0]
        # y = self.agent[1]
        # if policy_output == 0 and self.agent[1] == self.rows - 1: # going up with a jump, and not in air currently
        #     self.agent[1] += 1
        agent_old = copy.copy(self.agent)

        if policy_output == 0: # right
            self.agent[0] += 1

        if policy_output == 1: # left
            self.agent[0] -= 1

        if policy_output == 2: # down
            self.agent[1] += 1

        if policy_output == 3: # up
            self.agent[1] -= 1

        self.agent[0] = max(min(self.agent[0], self.cols-1), 0)
        self.agent[1] = max(min(self.agent[1], self.rows-1), 0)

        if tuple(self.agent) in self.targets:
            self.targets.remove(tuple(self.agent))
            reward_mod += 10

        self.screen[agent_old[1]][agent_old[0]] = 0
        self.screen[self.agent[1]][self.agent[0]] = 1


        if len(self.targets) == 0:
            return None, reward_mod
        else:
            return self.get_state(), -1 + reward_mod
    
    def generate_fresh(self):
        return Gobble(self.args)
    
    def plot(self):
        print(np.array(self.screen))


class RockOn:
    state_size = 6*6
    num_actions = 4

    def __init__(self, args):
        self.args = args
        self.rows = args.rows
        self.cols = args.cols

        # self.blockers = args.blockers # set of (x, y) arrays?
        # self.rocks = [t for t in args.rocks]
        
        self.agent = np.array([0, self.rows-1]) # bottom left location
        # self.agent_velocity = np.array([0, 0])

        self.screen = [[0 for i in range(self.cols)] for i in range(self.rows)]
        self.rocks = [[randint(0, self.cols-1), randint(-5,0)] for i in range(args.num_rocks)]
        # self.rock_movs_x = self.args.movs_x
        # self.rock_movs_y = self.args.movs_y

        # populate the screen image, with a 1 for the agent, -1 for blockers, 0 otherwise
        # self.agent_screen[self.agent[1]][self.agent[0]] = 0.1
        for rock in self.rocks:
            if rock[1] >= 0:
                self.screen[rock[1]][rock[0]] = -0.1
        
        self.prev_screen = copy.deepcopy(self.screen)
        self._t = 0
        # self.goal = np.array([self.cols - 1, self.rows - 1])

    def get_state(self):
        '''
        ret: image?
        '''
        l = []
        for r, rp in zip(self.screen, self.prev_screen):
            l += list(np.array(r) + 0.5*np.array(rp))
        return l
        # return [[self.screen, self.agent_screen]]

    def step(self, policy_output):
        '''
        input: int
        ret: state (list), reward (int)
        '''
        self.prev_screen = copy.deepcopy(self.screen)
        self._t += 1
        policy_output = policy_output.item()

        agent_old = copy.copy(self.agent)

        if policy_output == 0: # right
            self.agent[0] += 1

        if policy_output == 2: # left
            self.agent[0] -= 1

        # if policy_output == 2: # down
        #     self.agent[1] += 1

        # if policy_output == 3: # up
        #     self.agent[1] -= 1

        self.agent[0] = max(min(self.agent[0], self.cols-1), 0)
        # self.agent[1] = max(min(self.agent[1], self.rows-1), 0)

        # update rocks
        for i in range(len(self.rocks)):
            r_old = copy.copy(self.rocks[i])
            # r[0] = (r[0] + self.rock_movs_x[i][self._t]) % self.cols
            # r[1] = (r[1] + self.rock_movs_y[i][self._t]) % self.rows
            self.rocks[i][1] += 1
            if self.rocks[i][1] >= self.args.rows:
                self.rocks[i] = [randint(0, self.cols-1), randint(-5,0)]
            elif 0 <= self.rocks[i][1] < self.args.rows:
                self.screen[self.rocks[i][1]][self.rocks[i][0]] = -0.1
            if 0 <= r_old[1] < self.args.rows:
                self.screen[r_old[1]][r_old[0]] = 0

        # update agent
        self.screen[agent_old[1]][agent_old[0]] = 0
        self.screen[self.agent[1]][self.agent[0]] = 0.1

        # reward and next state
        if list(self.agent) in self.rocks:
            return None, -1000
        else:
            return self.get_state(), self._t/2
    
    def generate_fresh(self):
        return RockOn(self.args)

    def plot(self):
        print(np.array(self.screen))

class NoGobble:
    state_size = 6*6
    num_actions = 4

    def __init__(self, args):
        self.args = args
        self.rows = args.rows
        self.cols = args.cols

        # self.blockers = args.blockers # set of (x, y) arrays?
        self.targets = set([tuple(t) for t in args.targets])
        
        self.agent = np.array([0, self.rows-1]) # bottom left location
        # self.agent_velocity = np.array([0, 0])

        self.screen = [[0 for i in range(self.cols)] for i in range(self.rows)]

        # populate the screen image, with a 1 for the agent, -1 for blockers, 0 otherwise
        self.screen[self.agent[1]][self.agent[0]] = 1
        for target in self.targets:
            self.screen[target[1]][target[0]] = -1
        
        self.prev_screen = copy.deepcopy(self.screen)
        # self.goal = np.array([self.cols - 1, self.rows - 1])

    def get_state(self):
        '''
        ret: image?
        '''
        l = []
        for r, rp in zip(self.screen, self.prev_screen):
            l += list(np.array(r) + 0.5*np.array(rp))
        return l
        # return [[list(np.array(self.screen) + 0.5 * np.array(self.prev_screen))]]

    def step(self, policy_output):
        '''
        input: int
        ret: state (list), reward (int)
        '''
        policy_output = policy_output.item()
        self.prev_screen = copy.deepcopy(self.screen)
        # x = self.agent[0]
        # y = self.agent[1]
        # if policy_output == 0 and self.agent[1] == self.rows - 1: # going up with a jump, and not in air currently
        #     self.agent[1] += 1
        agent_old = copy.copy(self.agent)

        if policy_output == 0: # right
            self.agent[0] += 1

        if policy_output == 1: # left
            self.agent[0] -= 1

        if policy_output == 2: # down
            self.agent[1] += 1

        if policy_output == 3: # up
            self.agent[1] -= 1

        if 0 > self.agent[0] > self.cols - 1:
            return None, -100
        if 0 > self.agent[1] > self.rows - 1:
            return None, -100

        self.agent[0] = max(min(self.agent[0], self.cols-1), 0)
        self.agent[1] = max(min(self.agent[1], self.rows-1), 0)

        if tuple(self.agent) in self.targets:
            return None, -100

        self.screen[agent_old[1]][agent_old[0]] = 0
        self.screen[self.agent[1]][self.agent[0]] = 1

        return self.get_state(), 1
    
    def generate_fresh(self):
        return NoGobble(self.args)

    def plot(self):
        print(np.array(self.screen))

