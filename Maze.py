import numpy as np

from NeuralNetwork import NeuralNetwork

# Define the maze matrix and other variables

new_maze = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
])

# Define the starting and goal positions
start_pos = (1, 1)
goal_pos = (5, 5)

# Define the directions, move_agent, is_valid_position, has_reached_goal functions
# ...

directions = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}


def has_reached_goal(pos, goal_pos):  # Add goal_pos parameter
    return pos == goal_pos


def is_valid_position(pos, maze):  # Add maze parameter
    row, col = pos
    if row < 0 or row >= maze.shape[0] or col < 0 or col >= maze.shape[1]:
        return False
    return maze[row, col] == 0


def move_agent(pos, direction, maze):  # Add maze parameter
    row, col = pos
    row_offset, col_offset = directions[direction]
    new_row, new_col = row + row_offset, col + col_offset
    new_pos = (new_row, new_col)
    if is_valid_position(new_pos, maze):  # Pass maze parameter
        return new_pos
    else:
        return pos

def direction_to_one_hot(direction):
    direction_index = list(directions.keys()).index(direction)
    one_hot = np.zeros(len(directions)+1)
    one_hot[direction_index] = 1
    return one_hot

def generate_training_data(num_samples):
    input_data = []
    output_data = []

    for i in range(num_samples):
        # Generate a random maze
        maze = np.random.randint(0, 2, size=(7, 7))
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1

        # Randomize start and goal positions
        start_pos = (np.random.randint(1, maze.shape[0] - 1), np.random.randint(1, maze.shape[1] - 1))
        goal_pos = (np.random.randint(1, maze.shape[0] - 1), np.random.randint(1, maze.shape[1] - 1))

        # Find the optimal path using A* search
        open_list = [(start_pos, [], 0)]  # Add the initial cost (0) to the tuple
        closed_set = set()
        while open_list:
            current_pos, path, current_cost = open_list.pop(0)  # Extract the current_cost from the tuple
            if has_reached_goal(current_pos, goal_pos):
                break
            if current_pos not in closed_set:
                closed_set.add(current_pos)
                for direction in directions:
                    new_pos = move_agent(current_pos, direction, maze)
                    if new_pos not in closed_set:
                        new_path = path + [direction]
                        heuristic = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
                        cost = current_cost + 1 + heuristic  # Update the cost calculation
                        open_list.append((new_pos, new_path, current_cost + 1))  # Store the updated cost
            open_list.sort(key=lambda x: x[2] + abs(x[0][0] - goal_pos[0]) + abs(x[0][1] - goal_pos[1]))

        # Add each position in the path and the corresponding next move as input and output data
        for j in range(len(path) - 1):
            current_pos = path[j]
            next_move = direction_to_one_hot(path[j + 1])
            input_vector = np.array([*current_pos, *goal_pos])
            input_data.append(input_vector)
            output_data.append(next_move)

    # return np.array(input_data), np.array(output_data)




# num_samples = 1000
# input_data, output_data = generate_training_data(num_samples)
# input_size = 4  # Current position (2) + Goal position (2)
# hidden_sizes = [16]
# output_size = 4  # One-hot encoded directions (up, down, left, right)
# nn = NeuralNetwork(input_size, hidden_sizes, output_size)
#
# epochs = 100
# learning_rate = 0.001
# nn.train(input_data, output_data, epochs, learning_rate)


def solve_maze_with_nn(maze, start_pos, goal_pos, nn, max_steps=1000):
    current_pos = start_pos
    path = [current_pos]

    step = 0
    while not has_reached_goal(current_pos) and step < max_steps:
        input_vector = np.array([*current_pos, *goal_pos])
        predictions = nn.forward(input_vector)

        move_index = np.argmax(predictions)
        direction = list(directions.keys())[move_index]

        current_pos = move_agent(current_pos, direction)

        if current_pos not in path:  # To avoid cycles
            path.append(current_pos)
        step += 1

    if has_reached_goal(current_pos):
        print("Goal reached!")
        print("Path:", path)
    else:
        print("Failed to reach the goal.")
        print("Last position:", current_pos)


# solve_maze_with_nn(new_maze, start_pos, goal_pos, nn)
