import atexit
import functools
import math
import time
import numpy as np
from traitlets import Float, Instance, Integer, observe, Bool
from Adafruit_MotorHAT import Adafruit_MotorHAT
from traitlets.config.configurable import Configurable, SingletonConfigurable

from utils import get_logger


logger = get_logger(__name__, __file__)


def get_default_speed(method):
    """Decorator function that pass default speed as speed argument if speed is None"""
    def wrapper(self, speed=None, *args, **kwargs):
        if speed is None:
            speed = self.default_speed
        return method(self, speed, *args, **kwargs)
    return wrapper

def odometry_decorator(func):
    def wrapper(self, *args, **kwargs):
        distance, angle = self.odometry()
        func(self, *args, **kwargs)
        return distance, angle
    return wrapper


class Motor(Configurable):

    value   = Float()

    # config
    alpha   = Float(default_value=1.0).tag(config=True)
    beta    = Float(default_value=0.0).tag(config=True)

    def __init__(self, driver, channel, *args, move=True, **kwargs):
        super().__init__(*args, **kwargs)  # initializes traitlets

        self._driver = driver
        self._motor = self._driver.getMotor(channel)
        self.move = move
        if(channel == 1):
            self._ina = 1
            self._inb = 0
        else:
            self._ina = 2
            self._inb = 3
        atexit.register(self._release)

    @observe('value')
    def _observe_value(self, change):
        self._write_value(change['new'])

    def _write_value(self, value):
        """Maps and write (speed) value to motor driver"""
        if self.move:
            mapped_value = int(255.0 * (self.alpha * value + self.beta))
            speed = min(max(abs(mapped_value), 0), 255)
            self._motor.setSpeed(speed)
            if mapped_value < 0:
                self._motor.run(Adafruit_MotorHAT.FORWARD)
                self._driver._pwm.setPWM(self._ina,0,0)
                self._driver._pwm.setPWM(self._inb,0,speed*16)
            else:
                self._motor.run(Adafruit_MotorHAT.BACKWARD)
                self._driver._pwm.setPWM(self._ina,0,speed*16)
                self._driver._pwm.setPWM(self._inb,0,0)

    def _release(self):
        """Stops motor by releasing control"""
        self._motor.run(Adafruit_MotorHAT.RELEASE)
        self._driver._pwm.setPWM(self._ina,0,0)
        self._driver._pwm.setPWM(self._inb,0,0)


class Robot(SingletonConfigurable):

    left_motor  = Instance(Motor)
    right_motor = Instance(Motor)

    # For odometry
    odometry_last_update = 0
    wheel_circumference  = np.pi * 0.065
    wheelbase            = 0.12
    func_mps_gradient    = 1.5 # 1.8
    func_mps_intercept   = -0.041 # -0.04

    # config
    i2c_bus             = Integer(default_value=1).tag(config=True)
    left_motor_channel  = Integer(default_value=1).tag(config=True)
    left_motor_alpha    = Float(default_value=1.0).tag(config=True)
    right_motor_channel = Integer(default_value=2).tag(config=True)
    right_motor_alpha   = Float(default_value=1.0).tag(config=True)
    default_speed       = Float(default_value=0.14).tag(config=True)

    def __init__(self, *args, move=True, **kwargs):
        logger.debug("Robot class initialized")
        super(Robot, self).__init__(*args, **kwargs)
        self.motor_driver   = Adafruit_MotorHAT(i2c_bus=self.i2c_bus)
        self.left_motor     = Motor(self.motor_driver, channel=self.left_motor_channel, alpha=self.left_motor_alpha, move=move)
        self.right_motor    = Motor(self.motor_driver, channel=self.right_motor_channel, alpha=self.right_motor_alpha, move=move)

    @odometry_decorator
    def set_motors(self, left_speed, right_speed):
        self.left_motor.value   = left_speed
        self.right_motor.value  = right_speed
        logger.debug(f"Left motor: {left_speed}, right motor: {right_speed}")

    @get_default_speed
    @odometry_decorator
    def forward(self, speed=None):
        self.left_motor.value   = speed
        self.right_motor.value  = speed
        logger.debug(f"Left motor: {speed}, right motor: {speed}")

    @get_default_speed
    @odometry_decorator
    def backward(self, speed=None):
        self.left_motor.value   = -speed
        self.right_motor.value  = -speed
        logger.debug(f"Left motor: {-speed}, right motor: {-speed}")

    @get_default_speed
    @odometry_decorator
    def left(self, speed=None):
        self.left_motor.value   = -speed
        self.right_motor.value  = speed
        logger.debug(f"Left motor: {-speed}, right motor: {speed}")

    @get_default_speed
    @odometry_decorator
    def right(self, speed=None):
        self.left_motor.value   = speed
        self.right_motor.value  = -speed
        logger.debug(f"Left motor: {speed}, right motor: {-speed}")

    @odometry_decorator
    def stop(self):
        self.left_motor.value   = 0
        self.right_motor.value  = 0
        logger.debug(f"Left motor: 0, right motor: 0")

    def odometry(self):
        """Returns the change in position in meters and angle of the robot in radians"""
        # get delta time since last call
        now = time.time()

        if self.odometry_last_update:
            dt = now - self.odometry_last_update

            # get motor speeds
            left_speed = self.left_motor.value
            right_speed = self.right_motor.value

            # calculate distance traveled by each wheel
            left_distance = self.meter_per_second(left_speed) * dt
            right_distance = self.meter_per_second(right_speed) * dt

            # calculate change in angle and position
            delta_rad = (right_distance - left_distance) / self.wheelbase
            distance = (left_distance + right_distance) / 2.0

        else:

            distance, delta_rad = 0, 0

        self.odometry_last_update = now
        logger.debug(f"Traveled distance: {distance}, turned radians: {delta_rad}")
        return distance, delta_rad

    @classmethod
    def meter_per_second(cls, motor_speed):
        mps = max(np.abs(motor_speed) * cls.func_mps_gradient + cls.func_mps_intercept, 0)
        if 0 < motor_speed:
            return mps
        elif motor_speed < 0:
            return -mps
        else:
            return 0

