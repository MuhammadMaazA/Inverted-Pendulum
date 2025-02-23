#!/usr/bin/env python3
"""
PID-based cart-pole simulation using the single URDF 'cartpole.urdf':
 - rail->cart: prismatic along x
 - cart->rod: revolve about y
 - rod->bob: fixed

We measure the rod's angle each step, compute a PID force, 
and apply that force to the cart's prismatic joint. 
We run 20s or until user kills it. 
At t=2s we do a small tip & push. 
You can left-click to poke in the PyBullet GUI.
"""

import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- A simple PID Controller class ----------
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-100,100)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits

    def compute(self, measurement, dt):
        error = (self.setpoint - measurement)
        self.integral += error*dt
        derivative = (error - self.prev_error)/dt
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        self.prev_error = error
        # clamp
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        return output

def main():
    # PID Gains for rod angle
    # Tweak these to get stable performance.
    pid = PIDController(Kp=100.0, Ki=0.0, Kd=20.0, setpoint=0.0, output_limits=(-200,200))

    # Simulation time step
    dt = 1.0/240.0

    # Connect
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setRealTimeSimulation(0)

    # Load plane
    plane_id= p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=2.0)

    # Also load our custom URDF
    cartpole_id= p.loadURDF("cartpole.urdf", basePosition=[0,0,0], useFixedBase=False)

    # We have 3 joints: 
    #   0 => slider_to_cart (prismatic)
    #   1 => cart_to_rod (revolute)
    #   2 => rod_to_bob (fixed)
    # But let's verify or find them by name:
    nJ= p.getNumJoints(cartpole_id)
    prismatic_index= None
    revolve_index= None
    for j in range(nJ):
        info= p.getJointInfo(cartpole_id, j)
        jName= info[1].decode("utf-8")
        if jName=="slider_to_cart":
            prismatic_index= j
        elif jName=="cart_to_rod":
            revolve_index= j

    # Turn off default motor control
    for j in range(nJ):
        p.setJointMotorControl2(cartpole_id, j, controlMode=p.VELOCITY_CONTROL, force=0)

    # Let’s ensure initial angles=0 => rod upright
    p.resetJointState(cartpole_id, revolve_index, 0.0, 0.0)
    # Let’s ensure the cart is at x=0
    p.resetJointState(cartpole_id, prismatic_index, 0.0, 0.0)

    # Let it settle
    for _ in range(100):
        p.stepSimulation()

    # logs
    t_data= []
    x_data= []
    th_data= []
    f_data= []

    t_run= 0.0
    max_time= 20.0

    print("Running for 20s. Press Ctrl+C to exit earlier.")
    start_wall= time.time()

    try:
        while t_run< max_time:
            # measure cart x, rod angle
            # cart x => prismatic joint pos
            cart_js= p.getJointState(cartpole_id, prismatic_index)
            x= cart_js[0]
            xdot= cart_js[1]

            # rod angle => revolve joint pos
            rod_js= p.getJointState(cartpole_id, revolve_index)
            angle= rod_js[0]
            angledot= rod_js[1]

            # compute PID force on the rod angle
            force= pid.compute(angle, dt)

            # Disturb at t=2 => forcibly tip rod
            extra=0.0
            if abs(t_run-2.0)<0.5*dt:
                extra= 30.0
                # forcibly tip
                oldA= rod_js[0]
                newA= oldA + math.radians(5.0)
                p.resetJointState(cartpole_id, revolve_index, newA, 0.0)

            total_force= force + extra

            # apply to prismatic joint
            # we do "TORQUE_CONTROL" on the prismatic => that means a linear force in x
            # Actually for prismatic, PyBullet calls it "force" in the joint motor
            p.setJointMotorControl2(
                bodyUniqueId= cartpole_id,
                jointIndex= prismatic_index,
                controlMode= p.TORQUE_CONTROL,
                force= total_force
            )

            # step
            p.stepSimulation()

            # log
            t_data.append(t_run)
            x_data.append(x)
            th_data.append(angle)
            f_data.append(total_force)

            # sleep
            time.sleep(dt)
            t_run += dt
    except KeyboardInterrupt:
        print("User interrupted.")
    finally:
        # done
        p.disconnect()

        # plot
        plt.figure(figsize=(8,6))
        ax1= plt.subplot(3,1,1)
        ax1.plot(t_data, x_data, 'b-', label='Cart X(m)')
        ax1.grid(); ax1.legend()

        ax2= plt.subplot(3,1,2)
        ax2.plot(t_data, np.degrees(th_data), 'r-', label='Rod Angle (deg)')
        ax2.grid(); ax2.legend()

        ax3= plt.subplot(3,1,3)
        ax3.plot(t_data, f_data, 'm-', label='Control Force (N?)')
        ax3.grid(); ax3.legend()

        plt.suptitle("Cart-Pole PID, prismatic cart, revolve rod, fixed bob")
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    main()
