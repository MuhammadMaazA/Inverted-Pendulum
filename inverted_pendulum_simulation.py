#!/usr/bin/env python

import pybullet as p
import pybullet_data
import numpy as np
import math
import random
import time

# ------------- 1) GLOBAL SIMULATION SETTINGS -------------
SIMULATION_TIME = 10.0     # seconds
TIMESTEP        = 1./240.0 # PyBullet default or smaller
ENABLE_REALTIME = False    # If True, uses real-time stepping in PyBullet

# Disturbances
DISTURB_AT_TIME      = 2.0   # when to push the cart
DISTURB_FORCE_MAG    = 8.0   # Newtons
TIP_PEND_AT_TIME     = 4.0   # when to tip the pendulum
TIP_PEND_ANGLE_DEG   = 5.0   # how many degrees to instantly add to angle

# Noise levels (Gaussian std dev)
X_NOISE_STD        = 0.003
XDOT_NOISE_STD     = 0.003
THETA_NOISE_STD    = 0.002
THETADOT_NOISE_STD = 0.002


# ------------- 2) CONTROLLERS --------------------------
class PIDController:
    """
    Basic PID that tries to keep x=0, theta=0.
    Weighted error = (theta error) + 0.2*(x error).
    """
    def __init__(self, kp=50.0, ki=0.0, kd=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator = 0.0
        self.prev_error = 0.0

    def compute_force(self, state, dt):
        """
        state = [x, xdot, theta, thetadot] (filtered)
        """
        x, xdot, theta, thetadot = state

        # Weighted combined error
        err_theta = -theta
        err_x     = -x
        error = err_theta + 0.2 * err_x

        self.integrator += error * dt
        d_error = (error - self.prev_error) / dt
        self.prev_error = error

        F = self.kp * error + self.ki * self.integrator + self.kd * d_error
        return F


class PolePlacementController:
    """
    Full-state feedback: F = -K [x, xdot, theta, thetadot].
    Gains from a linear model around upright. Example numbers.
    """
    def __init__(self):
        self.K = np.array([-40.0, -10.0, 80.0, 15.0], dtype=float)

    def compute_force(self, state, dt):
        e = np.array(state) # [x, xdot, theta, thetadot]
        F = -self.K.dot(e)
        return F


class NonlinearController:
    """
    A partial feedback linearization or energy-based approach.
    F = (M+m)*g*sin(theta) - k1*xdot - k2*thetadot
    (m=0.05, M=0.5 default)
    """
    def __init__(self, m=0.05, M=0.5, g=9.81):
        self.m = m
        self.M = M
        self.g = g
        self.kx = 2.0     # cart velocity damping
        self.kth= 2.0     # pendulum angular velocity damping

    def compute_force(self, state, dt):
        x, xdot, theta, thetadot = state
        F = (self.M + self.m)*self.g*math.sin(theta) - self.kx*xdot - self.kth*thetadot
        return F


# ------------- 3) SENSOR NOISE + SIMPLE FILTER -------------
class Sensor:
    def __init__(self, alpha=0.9):
        # alpha = smoothing factor for exponential filter
        self.alpha = alpha
        self.prev_measure = None

    def measure_and_filter(self, raw_state):
        """
        raw_state = [x, xdot, theta, thetadot] from PyBullet
        Add noise, then exponential smoothing.
        """
        x_m     = raw_state[0] + random.gauss(0, X_NOISE_STD)
        xdot_m  = raw_state[1] + random.gauss(0, XDOT_NOISE_STD)
        th_m    = raw_state[2] + random.gauss(0, THETA_NOISE_STD)
        thdot_m = raw_state[3] + random.gauss(0, THETADOT_NOISE_STD)

        noisy = np.array([x_m, xdot_m, th_m, thdot_m])

        if self.prev_measure is None:
            self.prev_measure = noisy
            return noisy

        filtered = self.alpha * self.prev_measure + (1-self.alpha) * noisy
        self.prev_measure = filtered
        return filtered


# ------------- 4) MAIN SIMULATION LOGIC -----------------
def run_simulation(controller_type='PID'):
    """
    1) Connect to PyBullet (GUI).
    2) Load plane + cart_pole.urdf.
    3) Setup chosen controller & sensor.
    4) Loop for N steps => read state, add noise, compute force, apply disturbance.
    5) Return arrays for plotting.
    """

    # Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1 if ENABLE_REALTIME else 0)

    # Load plane + cart
    plane_id = p.loadURDF("plane.urdf")
    cartpole_id = p.loadURDF("cart_pole.urdf",
                             basePosition=[0,0,0],
                             useFixedBase=False)

    # optionally reduce friction
    p.changeDynamics(plane_id,  -1, lateralFriction=0.0)
    p.changeDynamics(cartpole_id, -1, lateralFriction=0.0)

    # Choose controller
    if controller_type.upper() == 'PID':
        controller = PIDController()
    elif controller_type.upper() == 'POLE':
        controller = PolePlacementController()
    elif controller_type.upper() == 'NONLINEAR':
        controller = NonlinearController()
    else:
        raise ValueError("Unknown controller type.")

    # Create sensor for noise+filter
    sensor = Sensor(alpha=0.9)

    pivot_joint_idx = 0

    # Optionally set an initial angle offset
    initial_angle_deg = 5.0
    p.resetJointState(cartpole_id, pivot_joint_idx,
                      initial_angle_deg*math.pi/180.0, 0.0)

    # We'll run for SIMULATION_TIME
    N = int(SIMULATION_TIME / TIMESTEP)

    t_data     = []
    x_data     = []
    theta_data = []
    force_data = []

    for i in range(N):
        t = i*TIMESTEP

        # read state from PyBullet
        cart_pos, cart_ori = p.getBasePositionAndOrientation(cartpole_id)
        cart_vel, cart_angvel = p.getBaseVelocity(cartpole_id)
        x    = cart_pos[0]
        xdot = cart_vel[0]

        joint_state = p.getJointState(cartpole_id, pivot_joint_idx)
        theta    = joint_state[0]
        thetadot = joint_state[1]

        # Disturbances
        disturb_force = 0.0
        if abs(t - DISTURB_AT_TIME) < 0.5*TIMESTEP:
            disturb_force += DISTURB_FORCE_MAG

        if abs(t - TIP_PEND_AT_TIME) < 0.5*TIMESTEP:
            # forcibly tip the pendulum angle
            new_theta = theta + math.radians(TIP_PEND_ANGLE_DEG)
            p.resetJointState(cartpole_id, pivot_joint_idx, new_theta, thetadot)
            theta = new_theta

        # measure state with noise & filter
        measured = sensor.measure_and_filter([x, xdot, theta, thetadot])

        # compute controller force
        F_control = controller.compute_force(measured, TIMESTEP)
        F_total   = F_control + disturb_force

        # apply force to cart
        p.applyExternalForce(cartpole_id, -1,
                             forceObj=[F_total,0,0],
                             posObj=cart_pos,
                             flags=p.WORLD_FRAME)

        # step simulation if not real-time
        if not ENABLE_REALTIME:
            p.stepSimulation()

        # log data
        t_data.append(t)
        x_data.append(x)
        theta_data.append(theta)
        force_data.append(F_total)

        # optionally slow loop to real-time speed
        time.sleep(TIMESTEP)

    p.disconnect()

    return np.array(t_data), np.array(x_data), np.array(theta_data), np.array(force_data)


def main():
    # Choose: 'PID', 'POLE', 'NONLINEAR'
    controller_type = 'PID'
    t, x, theta, force = run_simulation(controller_type=controller_type)

    # optional: plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.subplot(3,1,1)
        plt.plot(t, x, label='Cart Position (m)')
        plt.grid(); plt.legend()

        plt.subplot(3,1,2)
        plt.plot(t, np.degrees(theta), label='Pendulum Angle (deg)')
        plt.grid(); plt.legend()

        plt.subplot(3,1,3)
        plt.plot(t, force, label='Control Force (N)')
        plt.grid(); plt.legend()

        plt.suptitle(f'Controller = {controller_type}')
        plt.show()
    except ImportError:
        print("matplotlib not installed; skipping plots.")

    print("Simulation finished.")

if __name__ == "__main__":
    main()
