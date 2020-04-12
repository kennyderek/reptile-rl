from copy import copy

class WorldSimulator:

    def __init__(self):
        
        self.maze = []

        self.num_row = 10
        self.num_col = 10

        self.agent_x = 1
        self.agent_y = 1

        self.goal_x = 7
        self.goal_y = 3

        # generates an empty maze, W stands for a wall square
        top_bottom = ['W'] * self.num_col
        middle = ['W'] + [' '] * (self.num_col - 2) + ['W']
        self.maze.append(copy(top_bottom))
        for _ in range(self.num_row - 2):
            self.maze.append(copy(middle))
        self.maze.append(copy(top_bottom))
        self.maze[self.goal_y][self.goal_x] = 'G'
    
        # generates an information vector for each square
        self.maze_info = [[[] for c in range(self.num_col)] for r in range(self.num_col)]
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
                    goal_dir_encoding[1] = -1
                elif y > self.goal_y:
                    goal_dir_encoding[1] = 1

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
                self.maze_info[y][x] = goal_dir_encoding + wall_dist

    def __str__(self):
        s = ""
        for r in self.maze:
            s += "".join(r) + "\n"
        return s

    def move(self, action):
        '''
        action: 'N', 'S', 'E', or 'W' to move on the map
        return: reward (int)
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
            reward = 20
        else:
            reward = -1
        return reward

    def get_state(self):
        '''
        returns the maze info vector corresponding to the agent's current x, y position
        '''
        return self.maze_info[self.agent_y][self.agent_x]


# simple test case
world = WorldSimulator()
world.move('N')
print(world.agent_x, world.agent_y) # the agent should still be at (1, 1) since it hit a wall
world.move('S')
world.move('S')
print(world.get_state()) # in the goal row, so the y goal direction should be zero
for i in range(0, 6):
    print(world.move('E')) # last reward should be 20
print(world.get_state()) # at goal location, so the first two indices should both be zero