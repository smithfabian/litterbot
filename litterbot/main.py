import logging
from logging import handlers
from pathlib import Path
import time
from threading import Lock

from robotics import Robot
from camera import Camera, camera_config
from models import CollisionAvoidance, PathFinder
from traitlets import HasTraits, Float
from utils import Sliders
import numpy as np

from utils import create_grid, Node, astar, paint_line


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
file_log_handler = logging.FileHandler(log_dir / f"{__name__}.log")
# buffer_log_handler = logging.handlers.MemoryHandler(10, target=file_log_handler)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(log_formatter)
logger.addHandler(file_log_handler)
logger.propagate = False

ROWS, COLS = 100, 100
CELL_SIZE = 0.15 # 15cm


class CentralController(Sliders):
    def __init__(self, collision_avoidance=True, path_finder=True, **kwargs):
        super().__init__(**kwargs)
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
            self.x                          = 0 
            self.y                          = 0
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
                self.path = self._get_path_around_obsticle()
                self.robot.turn_to_node((self.x, self.y), self.theta, self.path.pop())
                self.on_motor_value_change()
                return # start over to check if path is blocked
            
            elif path_is_blocked and not self.has_path_finder:
                self.robot.left(speed=self.speed)
                self.on_motor_value_change()
            
            elif not path_is_blocked and self.path:
                # # Alternative implementation: Paint line on image and use line follower
                # image = paint_line(image, self.map, self.path, (self.x, self.y), self.theta)
                
                # robot is already turned towards next point
                speed_mps = self.robot.meter_per_second(self.robot.default_speed)
                t = CELL_SIZE / speed_mps
                self.robot.forward()
                self.on_motor_value_change(self.robot.left_motor.value, self.robot.right_motor.value)
                time.sleep(t)
                self.robot.stop()
                self.on_motor_value_change()
                # self.robot.turn_to_node((self.x, self.y), self.theta, self.path.pop())
                return

            else: # not path_is_blocked and self.has_path_finder:
                self.robot.forward(speed=self.speed)
                self.on_motor_value_change()

            if self.has_path_finder:
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

                print(f"Angle: {angle}, P: {P}, I: {I}, D: {D}, PID: {PID}, Steering: {steering}")

                self.robot.left_motor.value = max(min(min_speed + steering, 1.0), 0.0)
                self.robot.right_motor.value = max(min(min_speed - steering, 1.0), 0.0)

                if self.robot.left_motor.value >= 1.0 or self.robot.right_motor.value <= 0.0:
                    self.integral = 0 # anti-windup reset on integral error
        finally:
            self.process_lock.release()

    def _get_path_around_obsticle(self):
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
        
        if left_speed and right_speed:
            self.left_speed_old = left_speed
            self.right_speed_old = right_speed
            
        if self.last_update_time is not None:
            duration = current_time - self.last_update_time
            self.update_position(self.left_speed_old, self.right_speed_old, duration)
            
        if left_speed == 0 and right_speed == 0:
            self.last_update_time = None
        else:
            self.last_update_time = current_time

    def update_position(self, left_speed, right_speed, duration):
        left_speed  = self.robot.meter_per_second(left_speed)
        right_speed = self.robot.meter_per_second(right_speed)
        
        logger.debug(f"Updating position: left_speed={left_speed}, right_speed={right_speed}, duration={duration}")
        # Calculate the distance traveled by each wheel
        left_distance  = left_speed * duration
        right_distance = right_speed * duration

        # straight line motion
        if np.abs(left_distance - right_distance) < 1.0e-3:
            delta_theta = 0
            delta_x = left_distance * np.cos(self.theta)
            delta_y = left_distance * np.sin(self.theta)

        # pivot turn (oposite wheel directions and same speed)
        elif np.abs(left_distance + right_distance) < 1.0e-3:
            delta_theta = (right_distance - left_distance) / self.robot.wheelbase
            delta_x = 0
            delta_y = 0
        
        # complex turning motion
        else:
            delta_theta = (right_distance - left_distance) / self.robot.wheelbase
            R = (self.robot.wheelbase / 2) * ((left_distance + right_distance) / (left_distance - right_distance))
            delta_x =  R * np.sin(delta_theta + self.theta) - R * np.sin(self.theta)
            delta_y = -R * np.cos(delta_theta + self.theta) + R * np.cos(self.theta)

        # Update the robot's position and orientation
        self.x += delta_x # / CELL_SIZE
        self.y += delta_y # / CELL_SIZE
        self.theta = (self.theta + delta_theta) % (2 * np.pi)
        logger.debug(f"New position: x={self.x}, y={self.y}, theta={self.theta}")

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

    def cleanup(self):
        if getattr(self, "camera", None) is not None:
            self.camera.stop()


if __name__ == "__main__":
    cc = CentralController()
    cc.start()