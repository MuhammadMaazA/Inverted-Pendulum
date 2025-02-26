import pybullet as p
import pybullet_data
import time
import os
import math
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-10.0, 10.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.previous_error = 0.0
        self.output_limits = output_limits  # (min, max)

    def compute(self, measurement, dt):
        """
        Compute the PID control output and return (control_output, error).

        measurement: The current measurement (pole angle in radians).
        dt: Time step.
        """
        # Error: measured angle - setpoint (0 = upright)
        error = measurement - self.setpoint
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error

        # Clamp
        output = max(self.output_limits[0], min(output, self.output_limits[1]))
        return output, error

def main():
    # PID gains
    pid = PIDController(Kp=6.0, Ki=0.3, Kd=0.39, setpoint=0.0, output_limits=(-100.0, 100.0))
    dt = 1.0 / 240.0

    # Connect to PyBullet (GUI mode)
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Gravity and plane
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")

    # Load our new URDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "flagpole.urdf")
    cart_start_pos = [0, 0, 0.1]
    cart_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    cart_id = p.loadURDF(urdf_path, cart_start_pos, cart_start_orientation, useFixedBase=False)

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

    # Identify wheel joints and the pole joint
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
        print("Error: Could not find all wheel joints.")
        p.disconnect()
        return

    print("Wheel joint indices:", wheel_indices)
    print("Pole joint index:", pole_joint_index)

    # Initial tilt
    initial_tilt = 0.1
    p.resetJointState(cart_id, pole_joint_index, initial_tilt)

    # ---------------------------------
    # Add friction to the base and wheels
    # so the cart can roll properly.
    # ---------------------------------
    for link_idx in range(-1, num_joints):
        p.changeDynamics(
            cart_id, link_idx,
            lateralFriction=1.0,   # friction with ground
            rollingFriction=0.001,
            spinningFriction=0.001
        )

    # Data for plotting
    time_data = []
    error_data = []
    sim_time = 0.0
    max_sim_time = 10.0

    print(f"Simulation running for {max_sim_time} seconds.")
    try:
        while sim_time < max_sim_time:
            start_time = time.time()

            # Read the pole angle
            joint_state = p.getJointState(cart_id, pole_joint_index)
            pole_angle = joint_state[0]

            # Compute PID torque
            control_effort, error = pid.compute(pole_angle, dt)

            # Record
            error_data.append(error)
            time_data.append(sim_time)
            sim_time += dt

            print(f"Time: {sim_time:.2f}s, Angle: {pole_angle:.4f} rad, Error: {error:.4f}, Torque: {control_effort:.2f}")

            # Apply the same torque to all wheels
            for wj in wheel_indices:
                p.setJointMotorControl2(
                    bodyUniqueId=cart_id,
                    jointIndex=wj,
                    controlMode=p.TORQUE_CONTROL,
                    force=control_effort
                )

            # Optional: camera follow
            cart_pos, _ = p.getBasePositionAndOrientation(cart_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5,
                cameraYaw=50,
                cameraPitch=-30,
                cameraTargetPosition=cart_pos
            )

            p.stepSimulation()

            # Maintain real-time
            elapsed = time.time() - start_time
            if (dt - elapsed) > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        p.disconnect()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, error_data, label='Pole Angle Error (rad)', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('PID Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
