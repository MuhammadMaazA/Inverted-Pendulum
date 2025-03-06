import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, measurement, dt):
        error = measurement - self.setpoint
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        control_output = -(self.Kp * error + self.Ki * self.integral + self.Kd * derivative)
        return control_output
