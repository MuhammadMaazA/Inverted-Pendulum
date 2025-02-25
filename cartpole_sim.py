import pybullet as p
import pybullet_data
import time
import os
import math
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-10.0, 10.0), dead_zone=1e-4):
        """
        dead_zone: If abs(error) < dead_zone, treat error as 0
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.previous_error = 0.0
        self.output_limits = output_limits  # (min, max)
        self.dead_zone = dead_zone

    def compute(self, measurement, dt):
        """
        measurement: pole angle in radians
        dt: time step
        returns: (control_output, error)
        """
        # Error
        error = measurement - self.setpoint
        
        # Optional: dead zone to ignore tiny floating errors
        if abs(error) < self.dead_zone:
            error = 0.0

        # Integral
        self.integral += error * dt

        # Derivative
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        # PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Clamp
        output = max(self.output_limits[0], min(output, self.output_limits[1]))
        return output, error

def main():
    # PID gains:
    #  - Kp and Kd as before
    #  - Ki=0 to prevent slow integral drift
    pid = PIDController(Kp=6.0, Ki=0.0, Kd=0.39, setpoint=0.0, output_limits=(-100.0, 100.0))

    dt = 1.0 / 240.0

    # Connect to PyBullet (GUI)
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity and plane
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")

    # Load the URDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(current_dir)
    cart_id = p.loadURDF("flagpole.urdf", [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)

    # Disable default motor control
    num_joints = p.getNumJoints(cart_id)
    for j in range(num_joints):
        p.setJointMotorControl2(
            bodyUniqueId=cart_id,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0
        )

    # Identify joints
    wheel_names = ["tl_base_joint", "bl_base_joint", "tr_base_joint", "br_base_joint"]
    wheel_indices = []
    pole_joint_index = None
    for j in range(num_joints):
        info = p.getJointInfo(cart_id, j)
        joint_name = info[1].decode("utf-8")
        if joint_name in wheel_names:
            wheel_indices.append(j)
        elif joint_name == "pole_base_joint":
            pole_joint_index = j

    if pole_joint_index is None:
        print("Error: 'pole_base_joint' not found.")
        p.disconnect()
        return
    if len(wheel_indices) != 4:
        print("Error: Could not find all 4 wheels.")
        p.disconnect()
        return

    print("Wheel joints:", wheel_indices)
    print("Pole joint index:", pole_joint_index)

    # No initial tilt => angle=0 => no motion unless disturbed
    initial_tilt = 0.0
    p.resetJointState(cart_id, pole_joint_index, initial_tilt)

    # Add friction so small torques don't move the cart
    for link_idx in range(-1, num_joints):
        p.changeDynamics(
            cart_id, link_idx,
            lateralFriction=2.0,    # Higher friction
            rollingFriction=0.01,
            spinningFriction=0.01
        )

    time_data = []
    error_data = []
    sim_time = 0.0
    max_sim_time = 10.0

    print(f"Simulation running for {max_sim_time} seconds. The cart should remain still unless you disturb it.")
    try:
        while sim_time < max_sim_time:
            start_time = time.time()

            # Current pole angle
            pole_angle = p.getJointState(cart_id, pole_joint_index)[0]

            # Compute PID
            control_effort, error = pid.compute(pole_angle, dt)

            # Log data
            time_data.append(sim_time)
            error_data.append(error)
            sim_time += dt

            print(f"Time: {sim_time:.2f}s, Angle: {pole_angle:.4f} rad, Error: {error:.4f}, Torque: {control_effort:.2f}")

            # Apply torque to all wheels
            for wj in wheel_indices:
                p.setJointMotorControl2(
                    bodyUniqueId=cart_id,
                    jointIndex=wj,
                    controlMode=p.TORQUE_CONTROL,
                    force=control_effort
                )

            # Step simulation
            p.stepSimulation()

            # Keep real-time
            elapsed = time.time() - start_time
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        p.disconnect()

    # Plot the angle error
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, error_data, color='red', label='Pole Angle Error (rad)')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('PID Error Over Time (No Movement Unless Disturbed)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
