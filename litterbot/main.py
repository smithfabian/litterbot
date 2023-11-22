import time
from threading import Lock

from robotics import Robot
from camera import Camera, camera_config
from models import CollisionAvoidance, PathFinder
from traitlets import HasTraits, Float
from utils import Sliders
import numpy as np

from utils import create_grid, Node, astar, paint_line, get_logger


logger = get_logger(__name__, __file__)

ROWS, COLS = 100, 100
CELL_SIZE = 0.15 # 15cm


class CentralController(Sliders):
    def __init__(self, collision_avoidance=True, path_finder=True, **kwargs):
        super().__init__(**kwargs)
        logger.debug("CentralController class initializing")
        try:
            self.has_collision_avoidance    = collision_avoidance
            self.has_path_finder            = path_finder
            self.robot                      = Robot()
            self.camera                     = Camera(config=camera_config(fps=5))
            self.process_lock               = Lock()
            self.collision_avoidance        = CollisionAvoidance("data/collision_avoidance_2.pth", "alexnet")
            self.path_finder                = PathFinder("data/path_finder.pth", "resnet18")
            self.angle_last                 = 0.0
            self.integral_error             = 0.0
            self.map                        = np.zeros((ROWS, COLS), dtype=np.uint8)
            self.last_update_time           = None # For tracking distance and time
            # Initialize position and orientation
            self.left_speed_old             = None
            self.right_speed_old            = None
            self.x                          = 0 
            self.y                          = 0
            self.goal                       = None
            self.theta                      = 0  # Orientation angle in radians
            self.path                       = []

        except Exception:
            self.cleanup()
            raise

    def _execute(self, change):
        if not self.process_lock.acquire(blocking=False):
            # Process is already running, skip this frame
            return
        try:
            image = change['new']

            if self.has_collision_avoidance:
                path_is_blocked = self.collision_avoidance.is_blocked(image)
            else:
                path_is_blocked = False

            if path_is_blocked and self.has_path_finder:
                logger.debug("Path is blocked")
                self.path = self.get_path_around_obsticle()
                logger.debug(f"Found a new path around object: {self.path}")
                self.turn_to_node(self.path)  
                return # start over to check if path is blocked
            
            elif path_is_blocked and not self.has_path_finder:
                logger.debug("Path is blocked, turning left")
                self.robot.left()
                self.on_motor_value_change(left_speed=0, right_speed=self.robot.default_speed)
            
            elif not path_is_blocked and self.path:
                logger.debug(f"Driving to next A* path node")
                # # Alternative implementation: Paint line on image and use line follower
                # image = paint_line(image, self.map, self.path, (self.x, self.y), self.theta)
                
                # robot is already turned towards next point
                t = CELL_SIZE / self.robot.meter_per_second(self.robot.default_speed)
                self.robot.forward()
                self.on_motor_value_change(self.robot.default_speed, self.robot.default_speed)
                time.sleep(t)
                self.robot.stop()
                self.on_motor_value_change()
                self.turn_to_node(self.path)
                return

            elif not path_is_blocked and not self.has_path_finder:
                logger.debug("Free path, driving forward")
                self.robot.forward()

            else: # self.has_path_finder:
                logger.debug("Free path, folowing line")
                # These needs to be tuned
                P_gain = getattr(self, "p_gain", 0.1)
                I_gain = getattr(self, "i_gain", 0.0)
                D_gain = getattr(self, "d_gain", 0.8)
                min_speed = getattr(self, "speed", 0.1)
                steering_bias = getattr(self, "steering_bias", 0.0)

                angle = self.path_finder.run(image)
                angle = angle / np.pi # normalized_angle

                P = angle * P_gain

                self.integral_error += angle
                I = self.integral_error * I_gain

                D = (angle - self.angle_last) * D_gain
                self.angle_last = angle

                PID = P + I + D

                steering = PID + steering_bias

                logger.debug(f"Angle: {round(angle, 3)}, P: {P}, I: {I}, D: {D}, PID: {PID}, Steering: {steering}")
                
                left_speed = max(min(min_speed + steering, 1.0), 0.0)
                right_speed = max(min(min_speed - steering, 1.0), 0.0)
                self.robot.set_motors(left_speed, right_speed)
                self.on_motor_value_change(left_speed=left_speed, right_speed=right_speed)
                
                if self.robot.left_motor.value >= 1.0 or self.robot.right_motor.value <= 0.0:
                    self.integral = 0 # anti-windup reset on integral error
        finally:
            self.process_lock.release()

    def get_path_around_obsticle(self):
        logger.debug("Getting path around object")
        # Register obsticle
        obstacle_x = int(round(self.x + np.cos(self.theta)))
        obstacle_y = int(round(self.y + np.sin(self.theta)))
        if 0 <= obstacle_x < len(self.map) and 0 <= obstacle_y < len(self.map[0]):
            self.map[obstacle_x][obstacle_y] = 1

        if self.goal is None:
            # Assumes object is taking up 1 cell and path is directly behind it
            goal_x = int(round(self.x + 2 * np.cos(self.theta)))
            goal_y = int(round(self.y + 2 * np.sin(self.theta)))
            self.goal = (goal_x, goal_y) 
        
        if (obstacle_x, obstacle_y) == self.goal:
            self.goal = self.goal[0] + 2, self.goal[1] # TODO: make this more robust
        path = astar(Node((self.x, self.y)), Node((self.goal)), self.map)
        return path


    def on_motor_value_change(self, left_speed=None, right_speed=None):
        # might need to be called in asynchronously 
        current_time = time.time()
        
        if left_speed is not None and right_speed is not None:
            self.left_speed_old = left_speed
            self.right_speed_old = right_speed
            
        if self.last_update_time is not None:
            duration = current_time - self.last_update_time
            self._update_position(self.left_speed_old, self.right_speed_old, duration)
            
        if left_speed == 0 and right_speed == 0:
            self.last_update_time = None
        else:
            self.last_update_time = current_time

    def _update_position(self, left_speed, right_speed, duration):
        left_speed  = self.robot.meter_per_second(left_speed)
        right_speed = self.robot.meter_per_second(right_speed)
        
        logger.debug(f"Updating position: left_speed={round(left_speed, 2)}, right_speed={round(right_speed, 2)}, duration={round(duration, 2)}")
        # Calculate the distance traveled by each wheel
        left_distance  = left_speed * duration
        right_distance = right_speed * duration

        # straight line motion
        if np.abs(left_distance - right_distance) < 1.0e-3:
            logger.debug("Straight line motion")
            delta_theta = 0
            delta_x = left_distance * np.cos(self.theta)
            delta_y = left_distance * np.sin(self.theta)

        # pivot turn (oposite wheel directions and same speed)
        elif np.abs(left_distance + right_distance) < 1.0e-3:
            logger.debug("Pivot turn")
            delta_theta = (right_distance - left_distance) / self.robot.wheelbase
            delta_x = 0
            delta_y = 0
        
        # complex turning motion
        else:
            logger.debug("Complex turning motion")
            delta_theta = (right_distance - left_distance) / self.robot.wheelbase
            R = (self.robot.wheelbase / 2) * ((left_distance + right_distance) / (left_distance - right_distance))
            delta_x =  R * np.sin(delta_theta + self.theta) - R * np.sin(self.theta)
            delta_y = -R * np.cos(delta_theta + self.theta) + R * np.cos(self.theta)

        # Update the robot's position and orientation
        self.x += delta_x # / CELL_SIZE
        self.y += delta_y # / CELL_SIZE
        self.theta = (self.theta + delta_theta) % (2 * np.pi)
        logger.debug(f"New position: x={round(self.x, 2)}, y={round(self.y, 2)}, theta={round(self.theta, 3)}")

    def turn_to_node(self, node):
        try:
            node = self.path.pop()
            logger.debug(f"Turning to next A* path node: {node}")
        except IndexError:
            logger.debug(f"Reached the A* path goal node")
            self.goal = None
            return

        target_angle = np.arctan2(node[1] * CELL_SIZE - self.y, node[0] * CELL_SIZE - self.x)
        delta_angle = (target_angle - self.theta + np.pi) % (2 * np.pi) - np.pi
        
        # w=v/r where w:rad/sec v:meters/sec r:m
        angular_speed = self.robot.meter_per_second(self.robot.default_speed) / (self.robot.wheelbase / 2)
        t = abs(delta_angle) / angular_speed

        if delta_angle < 0:
            self.robot.right()
            self.on_motor_value_change(left_speed=self.robot.default_speed, right_speed=-self.robot.default_speed)
        else:
            self.robot.left()
            self.on_motor_value_change(left_speed=-self.robot.default_speed, right_speed=self.robot.default_speed)

        time.sleep(t)
        self.robot.forward()
        self.on_motor_value_change(left_speed=self.robot.default_speed, right_speed=self.robot.default_speed)

    def start(self):    
        self.camera.start()
        self.camera.observe(self._execute, names='value')

    def stop(self):
        try:
            self.camera.unobserve(self._execute, names='value')
        except ValueError:
            pass # Observer was not registered
        self.robot.stop()
        self.camera.stop()
        
        self.map              = np.zeros((ROWS, COLS), dtype=np.uint8)
        self.last_update_time = None # For tracking distance and time
        self.left_speed_old   = None
        self.right_speed_old  = None
        self.x                = 0 
        self.y                = 0
        self.goal             = None
        self.theta            = 0  # Orientation angle in radians
        self.path             = []

    def cleanup(self):
        if getattr(self, "camera", None) is not None:
            self.camera.stop()


if __name__ == "__main__":
    cc = CentralController()
    cc.start()