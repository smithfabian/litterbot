import time
from threading import Lock
from robotics import Robot
from camera import Camera, capture
from models import CollisionAvoidance, PathFinder
from traitlets import HasTraits, Float
from utils import Sliders
import numpy as np



class CentralController(Sliders):
    def __init__(self, collision_avoidance=True, path_finder=True, **kwargs):
        super().__init__(**kwargs)
        try:
            self.has_collision_avoidance    = collision_avoidance
            self.has_path_finder            = path_finder
            self.robot                  = Robot()
            self.camera                 = Camera(capture(816, 616))
            self.process_lock           = Lock()
            self.collision_avoidance    = CollisionAvoidance("data/collision_avoidance.pth", "alexnet")
            self.path_finder            = PathFinder("data/path_finder.pth", "resnet18")
            self.angle_last             = 0.0
            self.integral_error         = 0.0

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
                path_is_blocked = self.collision_avoidance.is_blocked(image) if self.collision_avoidance else False
            else:
                path_is_blocked = False

            if path_is_blocked and self.has_path_finder:
                self.robot.left(speed=self.speed)
                return
            elif not path_is_blocked and not self.has_path_finder:
                self.robot.forward(speed=self.speed)

            if self.has_path_finder:
                # These needs to be tuned
                P_gain = getattr(self, "p_gain", 0.08)
                I_gain = getattr(self, "i_gain", 0.0)
                D_gain = getattr(self, "d_gain", 0.0)
                min_speed = getattr(self, "speed", 0.08)
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