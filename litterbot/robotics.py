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


class Motor(Configurable):

    value   = Float()

    # config
    alpha   = Float(default_value=1.0).tag(config=True)
    beta    = Float(default_value=0.0).tag(config=True)

    def __init__(self, driver, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)  # initializes traitlets

        self._driver = driver
        self._motor = self._driver.getMotor(channel)
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
    movement_active     = Bool(default_value=False)
    wheel_circumference = Float(default_value=np.pi * 0.065)
    wheelbase           = Float(default_value=0.13)

    # config
    i2c_bus             = Integer(default_value=1).tag(config=True)
    left_motor_channel  = Integer(default_value=1).tag(config=True)
    left_motor_alpha    = Float(default_value=1.0).tag(config=True)
    right_motor_channel = Integer(default_value=2).tag(config=True)
    right_motor_alpha   = Float(default_value=1.0).tag(config=True)
    default_speed       = Float(default_value=0.2).tag(config=True)
    
    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        self.motor_driver   = Adafruit_MotorHAT(i2c_bus=self.i2c_bus)
        self.left_motor     = Motor(self.motor_driver, channel=self.left_motor_channel, alpha=self.left_motor_alpha)
        self.right_motor    = Motor(self.motor_driver, channel=self.right_motor_channel, alpha=self.right_motor_alpha)

    def set_motors(self, left_speed, right_speed):
        self.left_motor.value   = left_speed
        self.right_motor.value  = right_speed

    def forward(self, speed=default_speed):
        self.left_motor.value   = speed
        self.right_motor.value  = speed

    def backward(self, speed=default_speed):
        self.left_motor.value   = -speed
        self.right_motor.value  = -speed

    def left(self, speed=default_speed):
        self.left_motor.value   = -speed
        self.right_motor.value  = speed

    def right(self, speed=default_speed):
        self.left_motor.value   = speed
        self.right_motor.value  = -speed

    def stop(self):
        self.left_motor.value   = 0
        self.right_motor.value  = 0
    
    @staticmethod
    def meter_per_second(motor_speed):
        mps = max(np.abs(motor_speed) * 1.8 - 0.04, 0)
        if 0 < motor_speed:
            return mps
        elif motor_speed < 0:
            return -mps
        else:
            return 0

