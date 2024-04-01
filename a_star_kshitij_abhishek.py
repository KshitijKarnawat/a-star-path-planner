"""
a_star_kshitij_abhishek.py

@breif:     This module implements A-star algorithm for finding the shortest path in a graph.
@author:    Kshitij Karnawat, Abhishekh Reddy
@date:      19th March 2024
@version:   2.0

@github:    https://github.com/KshitijKarnawat/a-star-path-planner
"""

import numpy as np
import cv2 as cv
import time


class NewNode:
    """Class to represent a node in the graph
    """
    def __init__(self, pose, parent, cost_to_go, cost_to_come):
        """Initializes the node with its coordinates, parent and cost

        Args:
            coord (tuple): Coordinates of the node along with the angle
            parent (NewNode): Parent node of the current node
            cost_to_go (float): Cost to reach the current node
            cost_to_come (float): A-Star Hueristic for the current node (Eucledian Distance)
        """
        self.pose = pose
        self.parent = parent
        self.cost_to_go = cost_to_go
        self.cost_to_come = cost_to_come
        self.total_cost = cost_to_come + cost_to_go

# Reused from Previous Assignment
def create_map():
    """Generates the game map

    Returns:
        numpy array: A 2D array representing the game map
    """
    # Create map
    game_map = np.zeros((250, 600, 3), dtype=np.uint8)
    game_map.fill(255)

    # Create obstacles
    ### Refer https://docs.opencv.org/3.4/dc/da5/tutorial_py_drawing_functions.html on how to draw Polygons

    # Define rectangle vertices
    rectange_1 = np.array([[87, 50],
                           [87, 250],
                           [50, 250],
                           [50, 50]], dtype=np.int32)

    rectangle_2 = np.array([[175, 0],
                           [175, 200],
                           [137, 200],
                           [137, 0]], dtype=np.int32)

    # Define hexagon vertices
    side_length = 75
    hexagon_center = (325, 125)
    hexagon_vertices = []
    for i in range(6):
        angle_rad = np.deg2rad(90) + np.deg2rad(60 * i)  # Angle in radians for each vertex + 90 for rotating the hexagon
        x = int(hexagon_center[0] + side_length * np.cos(angle_rad))
        y = int(hexagon_center[1] - side_length * np.sin(angle_rad))
        hexagon_vertices.append([x, y])

    hexagon = np.array(hexagon_vertices, dtype=np.int32)

    # Define arch vertices
    arch = np.array([[550, 25],
                     [550, 225],
                     [450, 225],
                     [450, 187],
                     [510, 187],
                     [510, 62],
                     [450, 62],
                     [450, 25]], dtype=np.int32)

    game_map = cv.fillPoly(game_map, [rectange_1, rectangle_2, hexagon, arch], (0, 0, 0))

    game_map = cv.flip(game_map, 0)

    return game_map

# Reused from Previous Assignment
def in_obstacles(pose, clearance):
    """Checks if the given coordinates are in obstacles

    Args:
        coord (tuple): Coordinates to check

    Returns:
        bool: True if the coordinates are in obstacles, False otherwise
    """
    # Set Max and Min values for x and y
    x_max, y_max = 600, 250
    x_min, y_min = 0, 0

    x, y, heading = pose

    bloat = clearance
    vertical_shift = 224 # needed as hexagon center is made on x = 0

    if (x < x_min + bloat) or (x > x_max - bloat) or (y < y_min + bloat) or (y > y_max - bloat):
        return True

    # Rectangle 1
    elif (x >= 50 - bloat and x <= 87 + bloat) and (y >= 50 - bloat and y <= 250):
        return True

    # Rectangle 2
    elif (x >= 137 - bloat and x <= 175 + bloat) and (y >= 0 and y <= 200 + bloat):
        return True

    # Hexagon
    elif (x >= 260 - bloat) and (x <= 390 + bloat) and ((x  + 1.7333 * y) <= 465 - (2 * bloat) + vertical_shift ) and ((x - 1.7333 * y) >= 185 + (2 * bloat) - vertical_shift) and ((x - 1.7333 * y) <= 465 + bloat - vertical_shift ) and ((x  + 1.7333 * y) >= 185 - bloat + vertical_shift):
        return True

    # Arch
    # Divide the arch into 3 parts and check for each part

    # Part 1
    elif (x >= 510 - bloat and x <= 550 + bloat) and (y >= 25 + bloat and y <= 225 - bloat):
        return True

    # Part 2
    elif (x >= 450 - bloat and x <= 550 + bloat) and (y >= 187 - bloat and y <= 225 + bloat):
        return True

    # Part 3
    elif (x >= 450 - bloat and x <= 550 + bloat) and (y >= 25 - bloat and y <= 62 + bloat):
        return True

    return False

def near_goal(current_pose, goal_pose, threshold):
    x1, y1, _ = current_pose
    x2, y2, _ = goal_pose

    return np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2)) <= threshold

def calc_euclidian_distance(current_pose, goal_pose):
    """Calculates euclidian distance between the current and goal nodes
       for estimating cost to go

    Args:
        current_node_coord (tuple): Current node coordinate
        goal_node_coord (tuple): Goal node coordinate

    Returns:
        Float: Euclidian distance which is cost to move to the goal node
    """
    x1, y1, _ = current_pose
    x2, y2, _ = goal_pose

    return np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


def move_forward(L, node, goal_pose):
    x, y, heading = node.pose

    updated_x, updated_y = (x + (L * np.cos(np.deg2rad(heading))), y + (L * np.sin(np.deg2rad(heading))))

    cost_to_go = calc_euclidian_distance((updated_x, updated_y, heading), goal_pose)

    child = NewNode((int(round(updated_x, 0)), int(round(updated_y, 0)), heading), node, cost_to_go, node.cost_to_come + L)

    return child, L

def small_left_turn(L, node, goal_pose):
    x, y, heading = node.pose

    updated_heading = (heading + 30) % 360
    updated_x, updated_y = (x + (L * np.cos(np.deg2rad(updated_heading))), y + (L * np.sin(np.deg2rad(updated_heading))))

    cost_to_go = calc_euclidian_distance((updated_x, updated_y, updated_heading), goal_pose)

    child = NewNode((int(round(updated_x, 0)), int(round(updated_y, 0)), updated_heading), node, cost_to_go, node.cost_to_come + L)

    return child, L

def small_right_turn(L, node, goal_pose):
    x, y, heading = node.pose

    updated_heading = (heading - 30) % 360
    updated_x, updated_y = (x + (L * np.cos(np.deg2rad(updated_heading))), y + (L * np.sin(np.deg2rad(updated_heading))))

    cost_to_go = calc_euclidian_distance((updated_x, updated_y, updated_heading), goal_pose)

    child = NewNode((int(round(updated_x, 0)), int(round(updated_y, 0)), updated_heading), node, cost_to_go, node.cost_to_come + L)

    return child, L

def big_left_turn(L, node, goal_pose):
    x, y, heading = node.pose

    updated_heading = (heading + 60) % 360
    updated_x, updated_y = (x + (L * np.cos(np.deg2rad(updated_heading))), y + (L * np.sin(np.deg2rad(updated_heading))))

    cost_to_go = calc_euclidian_distance((updated_x, updated_y, updated_heading), goal_pose)

    child = NewNode((int(round(updated_x, 0)), int(round(updated_y, 0)), updated_heading), node, cost_to_go, node.cost_to_come + L)

    return child, L

def big_right_turn(L, node, goal_pose):
    x, y, heading = node.pose

    updated_heading = (heading - 60) % 360
    updated_x, updated_y = (x + (L * np.cos(np.deg2rad(updated_heading))), y + (L * np.sin(np.deg2rad(updated_heading))))

    cost_to_go = calc_euclidian_distance((updated_x, updated_y, updated_heading), goal_pose)

    child = NewNode((int(round(updated_x, 0)), int(round(updated_y, 0)), updated_heading), node, cost_to_go, node.cost_to_come + L)

    return child, L

def get_child_nodes(L, node, goal_pose, clearance):
    """Generates all possible child nodes for the given node

    Args:
        node (NewNode): Node to generate child nodes from
        goal_coord (tuple): Coordinates of the goal node

    Returns:
        list: List of child nodes and their costs
    """

    # child nodes list
    child_nodes = []

    # Create all possible child nodes
    child, child_cost = move_forward(L, node, goal_pose)
    if not in_obstacles(child.pose, clearance):
        child_nodes.append((child, child_cost))
    else:
        del child

    child, child_cost = small_left_turn(L, node, goal_pose)
    if not in_obstacles(child.pose, clearance):
        child_nodes.append((child, child_cost))
    else:
        del child

    child, child_cost = small_right_turn(L, node, goal_pose)
    if not in_obstacles(child.pose, clearance):
        child_nodes.append((child, child_cost))
    else:
        del child

    child, child_cost = big_left_turn(L, node, goal_pose)
    if not in_obstacles(child.pose, clearance):
        child_nodes.append((child, child_cost))
    else:
        del child

    child, child_cost = big_right_turn(L, node, goal_pose)
    if not in_obstacles(child.pose, clearance):
        child_nodes.append((child, child_cost))
    else:
        del child

    return child_nodes

def astar(L, start_pose, goal_pose, clearance):
    """Finds the shortest path from start to goal using Dijkstra's algorithm

    Args:
        start (tuple): Start coordinates
        goal (tuple): Goal coordinates

    Returns:
        list: A list of explored nodes
        list: A list of coordinates representing the shortest path
    """
    # Initialize open and closed lists
    open_list = []
    open_list_info = {}
    closed_list = []
    closed_list_info = {}
    path = []
    explored_nodes = []

    # Create start node and add it to open list
    start_node = NewNode(start_pose, None, calc_euclidian_distance(start_pose, goal_pose), 0)
    open_list.append((start_node, start_node.total_cost))
    open_list_info[start_node.pose] = start_node
    start_time = time.time()
    while open_list:
        # Get the node with the minimum total cost and add to closed list
        open_list.sort(key=lambda x: x[1]) # sort open list based on total cost
        current_node, _ = open_list.pop(0)
        cost_to_come = current_node.cost_to_come
        open_list_info.pop(current_node.pose)
        closed_list.append(current_node)
        closed_list_info[current_node.pose] = current_node

        # Check if goal reached
        if near_goal(current_node.pose, goal_pose, L // 2):
            end_time = time.time()
            print("Time taken by A-Star:", end_time - start_time)
            path = backtrack_path(current_node)
            return explored_nodes, path

        else:
            children = get_child_nodes(L, current_node, goal_pose, clearance)
            for child, child_cost in children:
                if child.pose in closed_list_info.keys():
                    del child
                    continue

                if child.pose in open_list_info.keys():
                    if child_cost + cost_to_come < open_list_info[child.pose].cost_to_come:
                        open_list_info[child.pose].cost_to_come = child_cost + cost_to_come
                        open_list_info[child.pose].total_cost = open_list_info[child.pose].cost_to_come + open_list_info[child.pose].cost_to_go
                        open_list_info[child.pose].parent = current_node
                else:
                    child.parent = current_node
                    open_list.append((child, child.total_cost))
                    open_list_info[child.pose] = child

                    explored_nodes.append(child)
    end_time = time.time()
    print("Time taken by A-Star:", end_time - start_time)
    return explored_nodes, None

# Reused from Previous Assignment
def backtrack_path(goal_node):
    """Backtracking algorithm for Dijkstra's algorithm

    Args:
        goal_node (NewNode): Goal node

    Returns:
        list: A list of coordinates representing the shortest path
    """
    path = []
    parent = goal_node
    while parent!= None:
        path.append((parent.pose[0], parent.pose[1]))
        parent = parent.parent
    return path[::-1]

# Reused from Previous Assignment
def vizualize(game_map, start, goal, path, explored_nodes):
    """Vizualizes the path and explored nodes

    Args:
        game_map (numpy array): A 2D array representing the game map
        start (tuple): Start coordinates
        goal (tuple): Goal coordinates
        path (list): A list of coordinates representing the shortest path
        explored_nodes (list): A list of explored nodes
    """
    start_time = time.time()

    cv.circle(game_map, (start[0], game_map.shape[0] - start[1] - 1), 5, (0, 0, 255), -1)
    cv.circle(game_map, (goal[0], game_map.shape[0] - goal[1] - 1), 5, (0, 255, 0), -1)

    game_video = cv.VideoWriter('game_vizualization.avi', cv.VideoWriter_fourcc('M','J','P','G'), 60, (600, 250))
    game_map_copy = game_map.copy()
    count = 0
    for node in explored_nodes:
        cv.arrowedLine(game_map, (node.parent.pose[0], game_map.shape[0] - node.parent.pose[1] - 1), (node.pose[0], game_map.shape[0] - node.pose[1] - 1), [100, 255, 100], 1, tipLength=0.5)
        cv.arrowedLine(game_map_copy, (node.parent.pose[0], game_map.shape[0] - node.parent.pose[1] - 1), (node.pose[0], game_map.shape[0] - node.pose[1] - 1), [100, 255, 100], 1, tipLength=0.5)

        count += 1
        if count == 100:
            game_video.write(game_map.astype(np.uint8))
            count = 0

    mid_time = time.time()
    print("Time taken to draw explored nodes:", mid_time - start_time)

    if path is not None:
        for i in range(0, len(path) - 1):
            cv.arrowedLine(game_map, (path[i][0], game_map.shape[0] - path[i][1] - 1), (path[i+1][0], game_map.shape[0] - path[i+1][1] - 1), [255, 0, 0], 1, tipLength=0.5)
            cv.arrowedLine(game_map_copy, (path[i][0], game_map.shape[0] - path[i][1] - 1), (path[i+1][0], game_map.shape[0] - path[i+1][1] - 1), [255, 0, 0], 1, tipLength=0.5)
            game_video.write(game_map.astype(np.uint8))

    cv.circle(game_map_copy, (start[0], game_map.shape[0] - start[1] - 1), 5, (0, 0, 255), 2)
    cv.circle(game_map_copy, (goal[0], game_map.shape[0] - goal[1] - 1), 5, (0, 255, 0), 2)
    cv.imwrite('final_map.png', game_map_copy)
    game_video.release()
    end_time = time.time()
    print("Time taken to draw path:", end_time - mid_time)

def main():
    game_map = create_map()

    # get start and end points from user
    start_point_input = (int(input("Enter x coordinate of start point: ")), int(input("Enter y coordinate of start point: ")), int(input("Enter the start angle of the robot in multiples of 30deg(0 <= theta <= 360): ")))
    goal_point_input = (int(input("Enter x coordinate of goal point: ")), int(input("Enter y coordinate of goal point: ")), int(input("Enter the goal angle of the robot in multiples of 30deg(0 <= theta <= 360): ")))
    clearance_input = int(input("Enter the clearance for robot: "))
    robot_radius = int(input("Enter robot radius: "))
    L = int(input("Enter the step length of the robot (1 <= L <= 10): "))

    # Convert input 1200x500 space to 600x250 space
    start_point = (int(np.interp(start_point_input[0], [0, 1200], [0, 600])), int(np.interp(start_point_input[1], [0, 500], [0, 250])), start_point_input[2])
    goal_point = (int(np.interp(goal_point_input[0], [0, 1200], [0, 600])), int(np.interp(goal_point_input[1], [0, 500], [0, 250])), goal_point_input[2])
    clearance = int(clearance_input + robot_radius // 2)

    # Check if start and goal points are in obstacles
    if in_obstacles(start_point, clearance):
        print("Start point is in obstacle")
        return

    if in_obstacles(goal_point, clearance):
        print("Goal point is in obstacle")
        return

    # find shortest path
    explored_nodes, shortest_path = astar(L, start_point, goal_point, clearance)
    if shortest_path == None:
        print("No path found")

    # visualize path
    vizualize(game_map, start_point, goal_point, shortest_path, explored_nodes)

    # show map
    cv.imshow('Map', game_map)

    # wait for key press
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
