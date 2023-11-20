import heapq
import time

import cv2
import numpy as np
from traitlets import HasTraits


def create_grid(rows, cols):
    """Create a grid of zeros."""

    return np.zeros((rows, cols), dtype=np.uint8)


class Node:
    """A class to represent a node in the A* algorithm."""
    
    __slots__ = ['position', 'parent', 'g', 'h', 'f', 'id']
    _counter = 0

    def __init__(self, position, parent=None, g=0, h=0):
        x, y = position
        self.position = (round(x), round(y))
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h
        self.id = Node._counter
        Node._counter += 1

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

    def update(self, new_parent, new_g, new_h):
        self.parent = new_parent
        self.g = new_g
        self.h = new_h
        self.f = new_g + new_h

def heuristic(node, goal):
    return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])

def astar(start, goal, grid):
    """A* algorithm implementation."""

    open_set = []
    heapq.heappush(open_set, (start.f, start.id, start))
    open_dict = {start.position: start}
    closed_set = set()

    while open_set:
        current_node = heapq.heappop(open_set)[2]
        open_dict.pop(current_node.position)

        if current_node.position == goal.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if 0 <= node_position[0] < grid.shape[0] and 0 <= node_position[1] < grid.shape[1] and grid[node_position] == 0 :
                temp_node = Node(node_position)  # Temporary node for heuristic calculation
                new_node = Node(node_position, current_node, current_node.g + 1, heuristic(temp_node, goal))
                children.append(new_node)

        for child in children:
            if child.position in closed_set:
                continue

            if child.position in open_dict:
                existing_node = open_dict[child.position]
                if child.g < existing_node.g:
                    existing_node.update(child.parent, child.g, child.h)
            else:
                heapq.heappush(open_set, (child.f, child.id, child))
                open_dict[child.position] = child

        closed_set.add(current_node.position)


def bgr8_to_jpeg(value, quality=75):
    """Convert a BGR8 image to JPEG"""
    return bytes(cv2.imencode('.jpg', value)[1])

def paint_line(img, grid, path, center, theta, color=(0, 0, 0), thickness=2):
    """Draw a line on an image."""
    x1, y1, x2, y2, x3, y3, x4, y4 = 0, 0, 0, 0, 0, 0, 0, 0
    # corners of grid top left, bottom left, bottom right, top right
    # TODO: should be corners of visible grid
    pts_grid = np.float32([[0, 0], [0, grid.shape[0]], [grid.shape[1], grid.shape[0]], [grid.shape[1], 0]]).reshape(-1, 1, 2)
    # corresponding corners of image
    pts_img = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).reshape(-1, 1, 2) # TODO: find these points
   
    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1)
    # Apply the rotation to the grid corners
    rotated_pts_grid = cv2.transform(np.array([pts_grid]), rotation_matrix)
   
    M = cv2.getPerspectiveTransform(rotated_pts_grid, pts_img)
    path = np.array(path, dtype=np.float32) # only use first 3 points
    path_transformed = cv2.perspectiveTransform(path.reshape(-1, 1, 2), M)
    for i in range(len(path_transformed) - 1):
        cv2.line(img, tuple(path_transformed[i][0]), tuple(path_transformed[i + 1][0]), color=color, thickness=thickness)
    
    return img


class Sliders(HasTraits):
    """A class to create sliders in notebook for the litterbot."""
    def __init__(self, **kwargs):
        super().__init__()
        
        for key, obj in kwargs.items():
            if isinstance(obj, HasTraits) and hasattr(obj, 'value'):
                setattr(self, key, obj.value)
                
                def _observer_callback(change, name=key):
                    setattr(self, name, change.new)
                
                obj.observe(_observer_callback, names='value')