# Importing all libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from queue import PriorityQueue
import time

# Creating class that creates objects with following attributes
class createNode:
    def __init__(self, pos, parent, c2c, tc):
        self.pos = pos
        self.parent = parent
        self.c2c = c2c
        self.tc = tc


def Round_05(x):
    return int(round(x, 0))

# Functions to move in the allowed directions
def Forward(L, node, goal_pos):
    actionCost = L
    x, y, th = node.pos
    x_new, y_new = (x + (L * np.cos(np.deg2rad(0))), y + (L * np.sin(np.deg2rad(0))))
    cgoal = np.linalg.norm(np.asarray((x_new, y_new)) - np.asarray((goal_pos[0], goal_pos[1])))
    child_created = createNode((Round_05(x_new), Round_05(y_new), th), node, node.c2c + actionCost,
                               node.c2c + actionCost + cgoal)
    return actionCost, cgoal, child_created


def Left30(L, node, goal_pos):
    actionCost = L
    x, y, th = node.pos
    x_new, y_new = (x + L * np.cos(np.deg2rad(30)), y + L * np.sin(np.deg2rad(30)))
    cgoal = np.linalg.norm(np.asarray((x_new, y_new)) - np.asarray((goal_pos[0], goal_pos[1])))
    child_created = createNode((Round_05(x_new), Round_05(y_new), (th + 30) % 360), node, node.c2c + actionCost,
                               node.c2c + actionCost + cgoal)
    return actionCost, cgoal, child_created


def Right30(L, node, goal_pos):
    actionCost = L
    x, y, th = node.pos
    x_new, y_new = (x + L * np.cos(np.deg2rad(-30)), y + L * np.sin(np.deg2rad(-30)))
    cgoal = np.linalg.norm(np.asarray((x_new, y_new)) - np.asarray((goal_pos[0], goal_pos[1])))
    child_created = createNode((Round_05(x_new), Round_05(y_new), (th - 30) % 360), node, node.c2c + actionCost,
                               node.c2c + actionCost + cgoal)
    return actionCost, cgoal, child_created


def Left60(L, node, goal_pos):
    actionCost = L
    x, y, th = node.pos
    x_new, y_new = (x + L * np.cos(np.deg2rad(60)), y + L * np.sin(np.deg2rad(60)))
    cgoal = np.linalg.norm(np.asarray((x_new, y_new)) - np.asarray((goal_pos[0], goal_pos[1])))
    child_created = createNode((Round_05(x_new), Round_05(y_new), (th + 60) % 360), node, node.c2c + actionCost,
                               node.c2c + actionCost + cgoal)
    return actionCost, cgoal, child_created


def Right60(L, node, goal_pos):
    actionCost = L
    x, y, th = node.pos
    x_new, y_new = (x + L * np.cos(np.deg2rad(-60)), y + L * np.sin(np.deg2rad(-60)))
    cgoal = np.linalg.norm(np.asarray((x_new, y_new)) - np.asarray((goal_pos[0], goal_pos[1])))
    child_created = createNode((Round_05(x_new), Round_05(y_new), (th - 60) % 360), node, node.c2c + actionCost,
                               node.c2c + actionCost + cgoal)
    return actionCost, cgoal, child_created


# Function to create obstacles in the visualization space
def Obstacle_space(dims):
    w, h = dims
    angle = np.deg2rad(30)
    grid = np.zeros((h + 1, w + 1, 3), dtype=np.uint8)
    grid.fill(255)
    rect1 = np.array([[100, 150], [150, 150], [150, 250], [100, 250]])
    rect2 = np.array([[100, 100], [150, 100], [150, 0], [100, 0]])
    triangle = np.array([[460, 25], [510, 125], [460, 225]])
    hexagon = np.array([[300, 200], [300 + 75 * np.cos(angle), 200 - 75 * np.sin(angle)],
                        [300 + 75 * np.cos(angle), 50 + 75 * np.sin(angle)], [300, 50],
                        [300 - 75 * np.cos(angle), 50 + 75 * np.sin(angle)],
                        [300 - 75 * np.cos(angle), 200 - 75 * np.sin(angle)]]).astype(int)
    grid = cv2.fillPoly(grid, pts=[rect1, rect2, triangle, hexagon], color=(0, 0, 0))
    grid = cv2.flip(grid, 0)
    return grid


# Function to check if the robot is in the obstacle space (including the clearance margin)
def invade_obstacle(loc, gap):
    xMax, yMax = [600 + 1, 250 + 1]
    xMin, yMin = [0, 0]
    x, y, th = loc

    h1 = (300, 200 + gap)
    h2 = (300 + (75 + gap) * np.cos(np.deg2rad(30)), 125 + (75 + gap) * np.sin(np.deg2rad(30)))
    h3 = (300 + (75 + gap) * np.cos(np.deg2rad(30)), 125 - (75 + gap) * np.sin(np.deg2rad(30)))
    h4 = (300, 50 - gap)
    h5 = (300 - (75 + gap) * np.cos(np.deg2rad(30)), 125 - (75 + gap) * np.sin(np.deg2rad(30)))
    h6 = (300 - (75 + gap) * np.cos(np.deg2rad(30)), 125 + (75 + gap) * np.sin(np.deg2rad(30)))

    lh1 = h1[1] + ((h2[1] - h1[1]) / (h2[0] - h1[0])) * (x - h1[0])
    lh2 = h2[0]
    lh3 = h4[1] + ((h3[1] - h4[1]) / (h3[0] - h4[0])) * (x - h4[0])
    lh4 = h5[1] + ((h4[1] - h5[1]) / (h4[0] - h5[0])) * (x - h5[0])
    lh5 = h5[0]
    lh6 = h6[1] + ((h1[1] - h6[1]) / (h1[0] - h6[0])) * (x - h6[0])

    t1 = (460 - gap // 2, 125 + (102 + 2.25 * gap))
    t2 = (511 + 1.12 * gap, 125)
    t3 = (460 - gap // 2, 125 - (102 + 2.25 * gap))

    lt1 = t1[0]
    lt2 = t1[1] + ((t2[1] - t1[1]) / (t2[0] - t1[0])) * (x - t1[0])
    lt3 = t2[1] + ((t3[1] - t2[1]) / (t3[0] - t2[0])) * (x - t2[0])

    if (y <= lh1) and (x <= lh2) and (y >= lh3) and (y >= lh4) and (x >= lh5) and (y <= lh6):
        return False

    if (x >= lt1) and (y <= lt2) and (y >= lt3):
        return False

    if (x < xMin + gap) or (y < yMin + gap) or (x >= xMax - gap) or (y >= yMax - gap):
        return False

    if ((x <= 150 + gap) and (x >= 100 - gap) and (y <= 100 + gap)) or (
            (x <= 150 + gap) and (x >= 100 - gap) and (y >= 150 - gap)):
        return False
    else:
        return True


# Function to create possible children of a parent node within given constraints
def get_child(node, L, goal_pos, gap):
    xMax, yMax = [600 + 1, 250 + 1]
    xMin, yMin = [0, 0]
    children = []

    # check if moving forward is possible
    (actionCost, cgoal, child) = Forward(L, node, goal_pos)
    if invade_obstacle(child.pos, gap):
        # if node is not generated, append in child list
        children.append((actionCost, cgoal, child))
    else:
        del child

        # check if moving left by 30 deg is possible
    (actionCost, cgoal, child) = Left30(L, node, goal_pos)
    if invade_obstacle(child.pos, gap):
        # if node is not generated, append in child list
        children.append((actionCost, cgoal, child))
    else:
        del child

    # check if moving left by 60 deg is possible
    (actionCost, cgoal, child) = Left60(L, node, goal_pos)
    if invade_obstacle(child.pos, gap):
        # if node is not generated, append in child list
        children.append((actionCost, cgoal, child))
    else:
        del child

    # check if moving right by 30 deg is possible
    (actionCost, cgoal, child) = Right30(L, node, goal_pos)
    if invade_obstacle(child.pos, gap):
        # if node is not generated, append in child list
        children.append((actionCost, cgoal, child))
    else:
        del child

    # check if moving right by 60 deg is possible
    (actionCost, cgoal, child) = Right60(L, node, goal_pos)
    if invade_obstacle(child.pos, gap):
        # if node is not generated, append in child list
        children.append((actionCost, cgoal, child))
    else:
        del child

    return children


# Function to create a backtrack path connecting all nodes resulting in shortest path by algorithm
def backtrack(current):
    path = []
    parent = current
    while parent != None:
        path.append(parent.pos)
        parent = parent.parent
    return path


# A star path planning algorithm
def Astar_algo(start_pos, goal_pos, L, gap):
    if not invade_obstacle(start_pos, gap):
        print("start_pos position is in Obstacle grid")
        return False
    if not invade_obstacle(goal_pos, gap):
        print("start_pos position is in Obstacle grid")
        return False
    grid = Obstacle_space((600, 250))
    tc_g = np.linalg.norm(np.asarray((start_pos[0], start_pos[1])) - np.asarray((goal_pos[0], goal_pos[1])))
    openList = []
    open_dict = {}
    closedList = []
    closed_dict = {}
    viz = []
    initial_node = createNode(start_pos, None, 0, tc_g)
    openList.append((initial_node.tc, initial_node))
    open_dict[initial_node.pos] = initial_node

    start = time.time()
    while len(openList) > 0:
        openList.sort(key=lambda x: x[0])
        currentCost, current = openList.pop(0)
        open_dict.pop(current.pos)
        closedList.append(current)
        closed_dict[current.pos] = current
        if np.linalg.norm(np.asarray(current.pos[:2]) - np.asarray(goal_pos[:2])) <= L * 0.5:
            pathTaken = backtrack(current)
            end = time.time()
            print('Time taken to execute algorithm in sec: ',(end - start))
            return (pathTaken, viz)
        else:
            childList = get_child(current, L, goal_pos, gap)
            for actionCost, actionGoal, child_created in childList:
                if child_created.pos in closed_dict:
                    del child_created
                    continue
                if child_created.pos in open_dict:
                    if open_dict[child_created.pos].c2c > current.c2c + actionCost:
                        open_dict[child_created.pos].parent = current
                        open_dict[child_created.pos].c2c = current.c2c + actionCost
                        open_dict[child_created.pos].tc = open_dict[child_created.pos].c2c + actionGoal
                else:
                    child_created.parent = current
                    child_created.c2c = current.c2c + actionCost
                    child_created.tc = child_created.c2c + actionGoal
                    openList.append((child_created.tc, child_created))
                    open_dict[child_created.pos] = child_created
                    x, y, th = child_created.pos
                    viz.append(child_created)


# Function to visualize the visited nodes and the shortest path using openCV
def visualization(viz, pathTaken, start_pos, goal_pos):
    save = cv2.VideoWriter('game_vizualization.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (1200//2, 500//2))
    grid = Obstacle_space((600, 250))
    cv2.circle(grid, (start_pos[0], grid.shape[0] - start_pos[1] - 1), 1, (0, 0, 255), 1)
    cv2.circle(grid, (goal_pos[0], grid.shape[0] - goal_pos[1] - 1), 1, (0, 0, 255), 1)
    start_time = time.time()
    for i in viz:
        start = i.parent.pos[:2]
        end = i.pos[:2]
        cv2.arrowedLine(grid, (start[0], grid.shape[0] - start[1] - 1), (end[0], grid.shape[0] - end[1] - 1),
                        [50, 200, 20], 1, tipLength=0.2)
        save.write(grid)

    for i in range(len(pathTaken[::-1]) - 1):
        x1, y1 = pathTaken[::-1][i][0], pathTaken[::-1][i][1]
        x2, y2 = pathTaken[::-1][i + 1][0], pathTaken[::-1][i + 1][1]
        cv2.arrowedLine(grid, (x1, grid.shape[0] - y1 - 1), (x2, grid.shape[0] - y2 - 1), [0, 0, 0], 1, tipLength=0.4)
        save.write(grid)
    save.write(grid)
    save.write(grid)
    save.write(grid)
    save.write(grid)
    save.release()
    end_time = time.time()
    print('Time taken to visualize in sec: ',(end_time - start_time))


# User input variables
xs= int(input('Enter start x-coordinate: '))
ys= int(input('Enter start y-coordinate: '))
ths= int(input('Enter start orientation in multiple of 30 deg: '))
start_pos= (xs,ys,ths)

xg= int(input('Enter goal x-coordinate: '))
yg= int(input('Enter goal y-coordinate: '))
thg= int(input('Enter goal orientation in multiple of 30 deg: '))
goal_pos= (xg,yg,thg)

L= int(input('Enter step length (1<= L <10): '))
gap = int(input('Enter clearance (robot radius + bloat): '))

pathTaken,viz = Astar_algo(start_pos,goal_pos,L,gap)
visualization(viz,pathTaken,start_pos,goal_pos)
