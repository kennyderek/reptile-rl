#code adapted from https://scipython.com/blog/making-a-maze/

import random

class Cell:
	'''
	Represents a single cell in a maze.
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


class Maze:
	'''Represents a maze as a grid of cells'''

	def __init__(self, num_rows, num_cols, start_row, start_col):
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.start_row = start_row
		self.start_col = start_col

		self.maze = [[Cell(row, col) for col in range(num_cols)] for row in range(num_rows)]

	def make_maze(self):
		total_cells = self.num_rows * self.num_cols

		stack = []
		current_cell = self.get_cell(self.start_row, self.start_col)

		current_cell_number = 1

		while current_cell_number < total_cells:
			neighbors = self.get_neighbors(current_cell)

			if not neighbors:
				current_cell = stack.pop()
				continue

			direction, next_cell = random.choice(neighbors)
			current_cell.remove_wall(next_cell, direction)
			stack.append(current_cell)
			current_cell = next_cell
			current_cell_number += 1

	def get_cell(self, row, col):
		'''Returns Cell object at (row, col)'''
		return self.maze[row][col]

	def get_neighbors(self, cell):
		'''Return list of unvisited neighbors to cell'''

		delta = [('W', (-1,0)),
                 ('E', (1,0)),
                 ('S', (0,1)),
                 ('N', (0,-1))]

		neighbors = []

		for direction, (dx, dy) in delta:
			next_row, next_col = cell.row + dx, cell.col + dy
			if (0 <= next_row < self.num_rows) and (0 <= next_col < self.num_cols):
				neighbor = self.get_cell(next_row, next_col)
				if neighbor.is_completely_surrounded():
					neighbors.append((direction, neighbor))
		return neighbors

	def __str__(self):
		'''Returns string representation of maze'''
		maze = ''
		for i in range(self.num_cols+1):
			maze += '__'
		maze += '\n'
		for row in range(self.num_rows):
			maze_row = "|"
			for col in range(self.num_cols):
				cell = self.get_cell(row, col)
				if cell.walls['E']:
					maze_row += (' |')
				elif cell.walls['S']:
					maze_row += ('__')
				else:
					maze_row += ('  ')
			maze_row += '\n'
			maze += maze_row
		for i in range(self.num_cols+1):
			maze += '__'
		return maze

	# def __str__(self):
	# 	'''Returns string representation of maze'''
	# 	maze_rows = ['-' * self.num_rows*2]
	# 	for col in range(self.num_cols):
	# 		maze_row = ['|']
	# 		for row in range(self.num_rows):
	# 			if self.maze[row][col].walls['E']:
	# 				maze_row.append(' |')
	# 			else:
	# 				maze_row.append('  ')
	# 		maze_rows.append(''.join(maze_row))
	# 		maze_row = ['|']
	# 		for row in range(self.num_rows):
	# 			if self.maze[row][col].walls['S']:
	# 				maze_row.append('--')
	# 			else:
	# 				maze_row.append(' +')
	# 		maze_rows.append(''.join(maze_row))
	# 	return '\n'.join(maze_rows)



# m = Maze(10, 10, 0, 0)
# m.make_maze()
# print (m)


