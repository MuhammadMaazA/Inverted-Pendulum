#!/usr/bin/env python
"""
inverted_pendulum_simulation.py

A full PyBullet-based inverted pendulum simulation, including:
- Nonlinear 3D physics (URDF).
- Sensor noise & basic filtering.
- Three controllers: PID, PolePlacement, Nonlinear (energy-based).
- Disturbances: push cart & tip pendulum.
- Real-time visualization in PyBullet + optional Matplotlib plots.

Requirements:
  pybullet, numpy, matplotlib, scipy
"""

import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random

# -- Simulation parameters --
SIM_TIME          = 10.0       # total sim time (s)
TIMESTEP          = 1./240.0   # PyBullet default
USE_REALTIME      = False      # if True => p.setRealTimeSimulation(1)
# Disturbances
DISTURB_TIME      = 2.0        # (s) when to apply push
DISTURB_FORCE     = 10.0       # (N) push magnitude
TIP_TIME          = 4.0        # (s) when to forcibly tip pendulum angle
TIP_ANGLE_DEG     = 5.0        # degrees offset
# Sensor noise
X_NOISE_STD       = 0.003
XDOT_NOISE_STD    = 0.003
THETA_NOISE_STD   = 0.002
THETADOT_NOISE_STD= 0.002


# ================================
#   CONTROLLERS
# ================================
class PIDController:
    def __init__(self, kp=50.0, ki=0.0, kd=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator = 0.0
        self.prev_error = 0.0

    def compute_force(self, state, dt):
        """
        state = [x, xdot, theta, thetadot]
        We'll do a combined error:
           error = thetaError + 0.2*xError
        to attempt to keep x=0, theta=0.
        Then standard PID on that error.
        """
        x, xdot, theta, thetadot = state
        e_theta = -theta    # want theta=0
        e_x     = -x        # want x=0
        error   = e_theta + 0.2*e_x

        self.integrator += error * dt
        d_error = (error - self.prev_error) / dt
        self.prev_error = error

        F = self.kp*error + self.ki*self.integrator + self.kd*d_error
        return F

class PolePlacementController:
    """
    Force = -K [x, xdot, theta, thetadot].
    Gains are from a linear approximation (hand-tuned or placed).
    """
    def __init__(self):
        # Example gain vector:
        self.K = np.array([-40.0, -10.0, 80.0, 15.0], dtype=float)

    def compute_force(self, state, dt):
        e = np.array(state)
        F = - self.K.dot(e)
        return F

class NonlinearController:
    """
    A simple partial feedback linearization or energy-based approach:
    F = (M+m)*g*sin(theta) - kx*xdot - kth*thetadot
    Adjust constants as needed.
    """
    def __init__(self, m=0.05, M=0.5, g=9.81):
        self.m = m
        self.M = M
        self.g = g
        self.kx = 2.0    # cart velocity damping
        self.kth= 2.0    # pendulum angular velocity damping

    def compute_force(self, state, dt):
        x, xdot, theta, thetadot = state
        F = (self.M+self.m)*self.g*math.sin(theta) - self.kx*xdot - self.kth*thetadot
        return F


# ================================
#   SENSOR NOISE & FILTER
# ================================
class Sensor:
    def __init__(self, alpha=0.9):
        """
        alpha = 0.9 => exponential filter factor (0 < alpha < 1)
        """
        self.alpha = alpha
        self.prev_measure = None

    def measure_and_filter(self, true_state):
        """
        Adds Gaussian noise to [x, xdot, theta, thetadot], then exponential smoothing.
        """
        x, xdot, th, thdot = true_state

        # Add noise
        xn     = x     + random.gauss(0, X_NOISE_STD)
        xdotn  = xdot  + random.gauss(0, XDOT_NOISE_STD)
        thn    = th    + random.gauss(0, THETA_NOISE_STD)
        thdotn = thdot + random.gauss(0, THETADOT_NOISE_STD)

        noisy = np.array([xn, xdotn, thn, thdotn])

        # Exponential filter
        if self.prev_measure is None:
            self.prev_measure = noisy
            return noisy
        filtered = self.alpha*self.prev_measure + (1.0 - self.alpha)*noisy
        self.prev_measure = filtered
        return filtered


# ================================
#   MAIN SIMULATION
# ================================
def run_simulation(controller_choice='PID'):
    # 1) Connect to PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1 if USE_REALTIME else 0)

    # 2) Load a plane and the cart+pole URDF
    plane_id = p.loadURDF("plane.urdf")
    cartpole_id = p.loadURDF("cart_pole.urdf", basePosition=[0,0,0], useFixedBase=False)

    # Optionally reduce friction to allow easy sliding
    p.changeDynamics(plane_id, -1, lateralFriction=0.0)
    p.changeDynamics(cartpole_id, -1, lateralFriction=0.0)

    pivot_joint = 0  # The only joint we have for the pendulum

    # 3) Create chosen controller
    if controller_choice.upper() == 'PID':
        controller = PIDController(kp=50, ki=0, kd=10)
    elif controller_choice.upper() == 'POLE':
        controller = PolePlacementController()
    elif controller_choice.upper() == 'NONLINEAR':
        controller = NonlinearController()
    else:
        raise ValueError("Unknown controller")

    # Create sensor for noise & filtering
    sensor = Sensor(alpha=0.9)

    # 4) Optional: set initial angle offset to see how it recovers
    initial_angle_deg = 5.0
    p.resetJointState(cartpole_id, pivot_joint, math.radians(initial_angle_deg), 0)

    # We'll run for SIM_TIME in increments of TIMESTEP
    steps = int(SIM_TIME / TIMESTEP)
    t_list, x_list, theta_list, force_list = [], [], [], []

    for i in range(steps):
        t = i*TIMESTEP

        # (a) Read the cart + pendulum state from PyBullet
        cart_pos, cart_ori = p.getBasePositionAndOrientation(cartpole_id)
        cart_vel, cart_ang_vel = p.getBaseVelocity(cartpole_id)
        x = cart_pos[0]
        xdot = cart_vel[0]

        joint_state = p.getJointState(cartpole_id, pivot_joint)
        theta = joint_state[0]
        thetadot = joint_state[1]

        # (b) Disturbances
        F_disturb = 0.0
        # push cart at DISTURB_TIME
        if abs(t - DISTURB_TIME) < 0.5*TIMESTEP:
            F_disturb += DISTURB_FORCE
        # tip pendulum at TIP_TIME
        if abs(t - TIP_TIME) < 0.5*TIMESTEP:
            new_angle = theta + math.radians(TIP_ANGLE_DEG)
            p.resetJointState(cartpole_id, pivot_joint, new_angle, thetadot)
            theta = new_angle

        # (c) Sensor noise + filter
        measured = sensor.measure_and_filter([x, xdot, theta, thetadot])

        # (d) Controller output force
        F_ctrl = controller.compute_force(measured, TIMESTEP)
        F_total = F_ctrl + F_disturb

        # (e) Apply the force on the cart
        p.applyExternalForce(cartpole_id, -1,
                             forceObj=[F_total,0,0],
                             posObj=cart_pos,
                             flags=p.WORLD_FRAME)

        # (f) Step simulation if not real-time
        if not USE_REALTIME:
            p.stepSimulation()

        # (g) Log data
        t_list.append(t)
        x_list.append(x)
        theta_list.append(theta)
        force_list.append(F_total)

        # Slow down the loop to match real time (approx)
        time.sleep(TIMESTEP)

    # Disconnect
    p.disconnect()

    return np.array(t_list), np.array(x_list), np.array(theta_list), np.array(force_list)

# ================================
#   MAIN
# ================================
def main():
    # Pick one: 'PID', 'POLE', or 'NONLINEAR'
    controller = 'PID'
    t, x, theta, force = run_simulation(controller)

    # Optional: plot results
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.subplot(3,1,1)
        plt.plot(t, x, label="Cart Position (m)")
        plt.grid(True); plt.legend()

        plt.subplot(3,1,2)
        plt.plot(t, np.degrees(theta), label="Pendulum Angle (deg)")
        plt.grid(True); plt.legend()

        plt.subplot(3,1,3)
        plt.plot(t, force, label="Control Force (N)")
        plt.grid(True); plt.legend()
        plt.xlabel("Time (s)")
        plt.suptitle(f"Controller = {controller}")
        plt.show()
    except ImportError:
        print("matplotlib not installed. No plots shown.")

    print("Simulation complete.")

if __name__ == "__main__":
    main()
