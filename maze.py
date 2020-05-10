#code adapted from https://scipython.com/blog/making-a-maze/

import random
from random import randint

'''
TODO: -hardcode connected walls for some cases
'''

class Cell:
	'''
	Represents a single cell in a maze, each with four walls.
	'''

	def __init__(self, row, col):
		self.row = row
		self.col = col
		self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

	def is_completely_surrounded(self):
		'''Returns whether cell has all walls'''
		return all(self.walls.values())

	def remove_wall(self, other_cell, wall):
		'''Remove corresponding wall between self and other cell'''
		self.walls[wall] = False
		if wall == 'N':
			other_wall = 'S'
		elif wall == 'S':
			other_wall = 'N'
		elif wall == 'E':
			other_wall = 'W'
		elif wall == 'W':
			other_wall = 'E'
		other_cell.walls[other_wall] = False



class Prelim_Maze:
	'''Represents a maze as a grid of cells'''

	def __init__(self, num_rows, num_cols, num_walls, start = None, goal = None):
		self.num_rows = num_rows
		self.num_cols = num_cols

		self.num_walls = num_walls
		self.num_interior_walls = 2*(self.num_rows)*(self.num_rows+1)-(self.num_rows*2 + self.num_cols*2)
		self.goal = goal

		self.maze = [[Cell(row, col) for col in range(num_cols)] for row in range(num_rows)] #maze of Cells

		if start is None:
			start_row = randint(0, num_rows-1)
			start_col = randint(0, num_cols-1)
			self.start = self.get_cell(start_row, start_col)
		else:
			self.start = self.get_cell(start[0], start[1])

		self.final_maze = None  #maze of Maze_Cells (every elements is different)

	def make_maze(self):
		'''Creates maze that is solvable'''

		total_cells = self.num_rows * self.num_cols

		stack = []
		current_cell = self.get_cell(self.start.row, self.start.col)

		current_cell_number = 1

		self.set_goal()

		while current_cell_number < total_cells:
			neighbors = self.get_neighbors(current_cell)

			if not neighbors:
				current_cell = stack.pop()
				continue

			direction, next_cell = random.choice(neighbors)
			current_cell.remove_wall(next_cell, direction)
			self.num_interior_walls -= 1
			stack.append(current_cell)
			current_cell = next_cell
			current_cell_number += 1

	def generate_random_cell(self):
		'''Returns random cell in maze grid'''
		random_row = random.randint(0, self.num_rows-1)
		random_col = random.randint(0, self.num_cols-1)

		return self.get_cell(random_row, random_col)

	def make_maze_with_num_walls(self):
		'''Generates maze with number of wall specified.  Prunes walls from solvable maze until desired number of walls is reached.'''

		self.make_maze()


		stack = []
		current_cell = self.generate_random_cell()

		goal_num_walls = self.num_walls

		while self.num_interior_walls > goal_num_walls:
			neighbors = self.get_all_neighbors(current_cell)

			if not neighbors:
				if len(stack) == 0:
					current_cell = self.generate_random_cell()
					continue
				else:
					current_cell = stack.pop()
					continue

			direction, next_cell = random.choice(neighbors)
			current_cell.remove_wall(next_cell, direction)
			self.num_interior_walls -= 1
			stack.append(current_cell)
			current_cell = next_cell


	def get_all_neighbors(self, cell):
		'''Returns all neighbors that are valid and still have any walls'''

		delta = [('N', (-1,0)),
                 ('S', (1,0)),
                 ('E', (0,1)),
                 ('W', (0,-1))]

		neighbors = []

		for direction, (dx, dy) in delta:
			next_row, next_col = cell.row + dx, cell.col + dy
			if (0 <= next_row < self.num_rows) and (0 <= next_col < self.num_cols):
				neighbor = self.get_cell(next_row, next_col)
				if cell.walls[direction]:
					neighbors.append((direction, neighbor))
		return neighbors


	def get_cell(self, row, col):
		'''Returns Cell object at (row, col)'''
		return self.maze[row][col]

	def get_maze_cell(self, row, col):
		'''Returns Maze Cell object at (row, col)'''
		return self.final_maze[2*row+1][2*col+1]

	def get_neighbors(self, cell):
		'''Return list of unvisited neighbors to cell (that are completely locked)'''

		delta = [('N', (-1,0)),
                 ('S', (1,0)),
                 ('E', (0,1)),
                 ('W', (0,-1))]

		neighbors = []

		row = cell.row
		col = cell.col

		for direction, (dx, dy) in delta:
			next_row, next_col = cell.row + dx, cell.col + dy
			if (0 <= next_row < self.num_rows) and (0 <= next_col < self.num_cols):
				neighbor = self.get_cell(next_row, next_col)
				if neighbor.is_completely_surrounded():
					neighbors.append((direction, neighbor))
		return neighbors

	def set_goal(self):
		'''Sets random cell in grid to be goal'''
		cell = self.generate_random_cell()

		while self.is_start(cell):
			cell = self.generate_random_cell()

		self.goal = cell
		return cell

	def is_start(self, cell):
		'''Returns whether cell is starting position'''
		return (cell.row == self.start.row and cell.col == self.start.col)

	def is_goal(self, cell):
		'''Returns whether cell is goal position'''
		return (cell.row == self.goal.row and cell.col == self.goal.col)


	def generate_maze(self):
		'''Generates overall maze of Maze_Cells with specified number of walls and random goal position'''
		# print (1)
		self.make_maze_with_num_walls()
		# print (2)
		if self.goal is None:
			self.set_goal()
		# print (3)
		self.generate_maze_of_maze_cells()
		# print (4)


	def print_preliminary_maze(self):
		'''Returns string representation of preliminary maze before converting to Maze Cells'''
		maze = ' '
		for i in range(self.num_cols):
			maze += '__ '
		maze += '\n'
		for row in range(self.num_rows):
			maze_row = "|"
			for col in range(self.num_cols):
				cell = self.get_cell(row, col)
				is_goal = self.is_goal(cell)
				if cell.walls['S'] and cell.walls['E'] and is_goal:
					maze_row += '_*|'
				else:
					if cell.walls['S'] and not cell.walls['E']:
						if is_goal:
							maze_row += '_*_ '
						else:
							maze_row += '__ '
					elif cell.walls['S']:
						if is_goal:
							maze_row += '_*_'
						else:
							maze_row += ('__')
					if not cell.walls['S'] and cell.walls['E']:
						if is_goal:
							maze_row += ' *|'
						else:
							maze_row += '  |'
					elif cell.walls['E']:
						if is_goal:
							maze_row += '*|'
						else:
							maze_row += ('|')
					if not cell.walls['S'] and not cell.walls['E']:
						if is_goal:
							maze_row += ' * '
						else:
							maze_row += ('   ')
			maze_row += '\n'
			maze += maze_row

		return maze

	def generate_maze_of_maze_cells(self):
		'''Converts maze of Cells to maze of Maze Cells'''

		num_rows = self.num_rows*2 + 1
		num_cols = self.num_cols*2 + 1

		new_maze = [[Maze_Cell(row, col) for col in range(num_cols)] for row in range(num_rows)]

		for row in range(self.num_rows):
			for col in range(self.num_cols):
				cell = self.get_cell(row, col)
				row = cell.row
				col = cell.col
				walls = cell.walls

				new_row = 2*row + 1
				new_col = 2*col + 1

				if row == self.start.row and col == self.start.col:
					new_maze[new_row][new_col].set_start()
				elif row == self.goal.row and col == self.goal.col:
					new_maze[new_row][new_col].set_goal()


				north = (new_row-1, new_col)
				south = (new_row+1, new_col)
				east = (new_row, new_col+1)
				west = (new_row, new_col-1)

				if walls['N'] and (0<=north[0]<num_rows and 0<=north[1]<num_cols):
					new_maze[north[0]][north[1]].set_wall()

				if walls['S'] and (0<=south[0]<num_rows and 0<=south[1]<num_cols):
					new_maze[south[0]][south[1]].set_wall()

				if walls['E'] and (0<=east[0]<num_rows and 0<=east[1]<num_cols):
					new_maze[east[0]][east[1]].set_wall()

				if walls['W'] and (0<=west[0]<num_rows and 0<=west[1]<num_cols):
					new_maze[west[0]][west[1]].set_wall()


			for col in range(num_cols):
				new_maze[0][col].set_wall()
				new_maze[num_rows-1][col].set_wall()

			for row in range(num_rows):
				new_maze[row][0].set_wall()
				new_maze[num_cols-1][col].set_wall()

			self.final_maze = new_maze

	def __str__(self):
		'''Returns string representation of maze'''
		num_rows = 2*self.num_rows+1
		num_cols = 2*self.num_cols+1

		new_maze = [[0 for col in range(num_cols)] for row in range(num_rows)]

		for new_row in range(num_rows):
			for new_col in range(num_cols):
				cell = self.final_maze[new_row][new_col]
				row = cell.row
				col = cell.col

				new_maze[new_row][new_col] = str(cell)

		str_new_maze = ''
		for row in new_maze:
			str_new_maze += str(row)
			str_new_maze += '\n'

		return str_new_maze




class Maze_Cell:
	'''
	Represents a single cell in a maze, treating walls as separate cells
	'''

	def __init__(self, row, col):
		self.row = row
		self.col = col
		self.wall_status = False
		self.start_status = False
		self.goal_status = False

	def is_wall(self):
		return self.wall_status

	def is_start(self):
		return self.start_status

	def is_goal(self):
		return self.goal_status

	def set_goal(self):
		self.goal_status = True

	def set_wall(self):
		self.wall_status = True

	def set_start(self):
		self.start_status = True


	def __str__(self):
		if self.is_goal():
			return "G"
		if self.is_wall():
			return "W"
		return " "

class Maze:
	'''Represents a maze as a grid of cells, treating each object as separate cell'''

	def __init__(self, num_rows, num_cols, num_walls, start = None, goal = None):
		# print ("HERE")
		prelim_maze = Prelim_Maze(num_rows, num_cols, num_walls, start, goal = None)
		# print ("ONE")
		prelim_maze.generate_maze()
		# print ("TWO")


		self.num_rows = 2*num_rows + 1
		self.num_cols = 2*num_cols + 1

		self.num_walls = num_walls

		self.maze = prelim_maze.final_maze

		if goal is None:
			self.goal = prelim_maze.goal
		else:
			self.goal = self.get_cell(2*goal[0]+1, 2*goal[1]+1)

		if start is None:
			self.start = prelim_maze.start
		else:
			self.start = self.get_cell(2*start[0]+1, 2*start[1]+1)


	def get_cell(self, row, col):
		'''Returns Maze_Cell object at (row, col)'''

		return self.maze[row][col]

	def get_array_maze(self):
		new_maze = [[0 for col in range(self.num_cols)] for row in range(self.num_rows)]

		for new_row in range(self.num_rows):
			for new_col in range(self.num_cols):
				cell = self.maze[new_row][new_col]
				row = cell.row
				col = cell.col

				new_maze[new_row][new_col] = str(cell)
		return new_maze

	def __str__(self):
		'''Returns string representation of maze'''
		new_maze = [[0 for col in range(self.num_cols)] for row in range(self.num_rows)]

		for new_row in range(self.num_rows):
			for new_col in range(self.num_cols):
				cell = self.maze[new_row][new_col]
				row = cell.row
				col = cell.col

				new_maze[new_row][new_col] = str(cell)

		str_new_maze = ''
		for row in new_maze:
			str_new_maze += str(row)
			str_new_maze += '\n'

		return str_new_maze



# if __name__ == "__main__":
# 	m = Maze(2, 2, 5, (0, 0))
# 	# m.generate_maze()
# 	# prelim = m.print_preliminary_maze()
# 	# print (prelim)
# 	# print (m.goal.row, m.goal.col)
# 	print (m)
