import math
import time
from threading import Lock
from robotics import Robot
from camera import Camera, camera_config
from models import CollisionAvoidance, PathFinder
from traitlets import HasTraits, Float
from utils import Sliders
import numpy as np

from utils import create_grid, Node, astar, paint_line

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

            self.robot.left_motor.observe(self.on_motor_value_change, names='value')
            self.robot.right_motor.observe(self.on_motor_value_change, names='value')

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
                return # start over to check if path is blocked
            
            elif path_is_blocked and not self.has_path_finder:
                self.robot.left(speed=self.speed)
            
            elif not path_is_blocked and self.path:
                # # Alternative implementation: Paint line on image and use line follower
                # image = paint_line(image, self.map, self.path, (self.x, self.y), self.theta)
                
                # robot is already turned towards next point
                speed_mps = self.robot.default_speed * 0.1 - 0.1 # m/s
                t = CELL_SIZE / speed_mps
                self.robot.forward(duration=t)
                # self.robot.turn_to_node((self.x, self.y), self.theta, self.path.pop())
                return

            else: # not path_is_blocked and self.has_path_finder:
                self.robot.forward(speed=self.speed)

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
        obstacle_x = int(round(self.x + math.cos(self.theta)))
        obstacle_y = int(round(self.y + math.sin(self.theta)))
        if 0 <= obstacle_x < len(self.map) and 0 <= obstacle_y < len(self.map[0]):
            self.map[obstacle_x][obstacle_y] = 1

        if self.goal is None:
            # Assumes object is taking up 1 cell and path is directly behind it
            goal_x = int(round(self.x + 2 * math.cos(self.theta)))
            goal_y = int(round(self.y + 2 * math.sin(self.theta)))
            self.goal = (goal_x, goal_y) 
        
        if (obstacle_x, obstacle_y) == self.goal:
            self.goal = self.goal[0] + 2, self.goal[1] # TODO: make this more robust
        path = astar(Node((self.x, self.y)), Node((self.goal)), self.map)
        return path


    def on_motor_value_change(self, change):
        # might need to be called in asynchronously 
        current_time = time.time()
        if self.last_update_time is not None:
            duration = current_time - self.last_update_time
            # left_speed = change[new] if change['owner'] == self.robot.left_motor
            # right_speed = change[new] if change['owner'] == self.robot.right_motor
            self.update_position(self.robot.left_motor.value, self.robot.right_motor.value, duration)
        self.last_update_time = current_time

    def update_position(self, left_speed, right_speed, duration):
        # Calculate the distance traveled by each wheel
        left_distance = self.robot.wheel_circumference * left_speed * duration
        right_distance = self.robot.wheel_circumference * right_speed * duration
        delta_theta = (right_distance - left_distance) / self.robot.wheelbase

        # straight line motion
        if left_speed == right_speed:
            delta_x = left_distance * math.cos(self.theta)
            delta_y = left_distance * math.sin(self.theta)
            delta_theta = 0
            

        # pivot turn (oposite wheel directions and same speed)
        elif left_speed == -right_speed:
            delta_x = 0
            delta_y = 0
            # TODO: delta_theta correct?
        
        # complex turning motion (opposite wheel directions and different speeds)
        elif left_speed != -right_speed and ((left_speed > 0) ^ (right_speed > 0)):
            # Calculate the radius from the ICR to the midpoint of the wheelbase
            R = (self.robot.wheelbase / 2) * ((left_speed + right_speed) / (left_speed - right_speed))

            # Angle traversed by robot (in radians)
            angle = (right_distance - left_distance) / self.robot.wheelbase

            # Calculate the change in x and y
            delta_x = R * (math.sin(self.theta + angle) - math.sin(self.theta))
            delta_y = -R * (math.cos(self.theta + angle) - math.cos(self.theta))

            # Update the robot's orientation
            self.theta += angle
            # TODO: delta_theta correct?

        # simple turning motion (same wheel directions and different speeds)
        else:
            # Calculate the change in orientation
            average_theta = self.theta + delta_theta / 2 # Average orientation during the motion
            delta_x = left_distance * math.cos(average_theta)
            delta_y = left_distance * math.sin(average_theta)

        # Update the robot's position and orientation
        self.x += delta_x / CELL_SIZE
        self.y += delta_y / CELL_SIZE
        self.theta = (self.theta + delta_theta) % (2 * np.pi)  # Keep theta within [0, 2π]

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