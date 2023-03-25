import numpy as np

from Maze import solve_maze_with_nn, direction_to_one_hot, has_reached_goal, move_agent, is_valid_position
from NeuralNetwork import NeuralNetwork


def create_maze():
    return np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ])


def get_start_and_goal_positions():
    return (1, 1), (5, 5)


def get_directions():
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }

def is_surrounded_by_walls(pos, maze):
    row, col = pos
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for offset in offsets:
        new_row, new_col = row + offset[0], col + offset[1]
        if is_valid_position((new_row, new_col), maze):
            return False
    return True

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

        print("maze: \n" , maze)

        # Randomize start and goal positions
        while True:
            start_pos = (np.random.randint(1, maze.shape[0] - 1), np.random.randint(1, maze.shape[1] - 1))
            if maze[start_pos] == 0 and not is_surrounded_by_walls(start_pos, maze):
                break

        while True:
            goal_pos = (np.random.randint(1, maze.shape[0] - 1), np.random.randint(1, maze.shape[1] - 1))
            if maze[goal_pos] == 0 and goal_pos != start_pos and not is_surrounded_by_walls(goal_pos, maze):
                break
        print("Start: ", start_pos)
        print("goal: ", goal_pos)

        # Find the optimal path using A* search
        open_list = [(start_pos, [], 0)]  # Add the initial cost (0) to the tuple
        print("open list: ", open_list)
        # print("open list[0]: ", open_list.pop(0))
        closed_set = set()
        while open_list:
            # Extract the current_cost from the tuple
            (current_pos, path, current_cost) = open_list.pop(0)
            # current_pos = open_list.pop(0)[0]
            # path = open_list.pop(0)[1]
            # current_cost = open_list.pop(0)[2]
            print("current_pos: ", current_pos)
            print("path: ", path)
            print("current_cost: ", current_cost)
            if has_reached_goal(current_pos, goal_pos):
                break
            if current_pos not in closed_set:
                closed_set.add(current_pos)
                for direction in get_directions():
                    new_pos = move_agent(current_pos, direction, maze)
                    if new_pos not in closed_set:
                        new_path = path + [direction]
                        heuristic = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
                        cost = current_cost + 1 + heuristic  # Update the cost calculation
                        open_list.append((new_pos, new_path, current_cost + 1))  # Store the updated cost
            open_list.sort(key=lambda x: x[2] + abs(x[0][0] - goal_pos[0]) + abs(x[0][1] - goal_pos[1]))
            print("open_list_sorted: ", open_list)

            # Add each position in the path and the corresponding next move as input and output data
            for j in range(len(path) - 1):
                current_pos = path[j]
                print("current pos", current_pos)
                next_move = direction_to_one_hot(path[j + 1])
                print("next move", current_pos)
                input_vector = np.array([*current_pos, *goal_pos])
                input_data.append(input_vector)
                output_data.append(next_move)
                print("inp: " , input_data)
                print("-------------")
                print("o/p: " ,output_data)

    return np.array(input_data), np.array(output_data)


def main():
    maze = create_maze()
    start_pos, goal_pos = get_start_and_goal_positions()
    directions = get_directions()

    # Train the neural network
    num_samples = 1
    input_data, output_data = generate_training_data(num_samples)
    input_size = 4  # Current position (2) + Goal position (2)
    hidden_sizes = [16]
    output_size = 4  # One-hot encoded directions (up, down, left, right)
    nn = NeuralNetwork(input_size, hidden_sizes, output_size)

    epochs = 100
    learning_rate = 0.001
    nn.train(input_data, output_data, epochs, learning_rate)

    # Solve the maze using the trained neural network
    solve_maze_with_nn(maze, start_pos, goal_pos, nn, directions)


if __name__ == "__main__":
    main()
