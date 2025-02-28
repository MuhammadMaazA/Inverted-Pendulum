import pybullet as p
import pybullet_data
import time
import os
import math
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-50.0, 50.0), dead_zone=1e-4):
        """
        dead_zone: if abs(error) < dead_zone, treat error as 0.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.previous_error = 0.0
        self.output_limits = output_limits
        self.dead_zone = dead_zone

    def compute(self, measurement, dt):
        # Compute error between measured value and desired setpoint
        error = measurement - self.setpoint
        if abs(error) < self.dead_zone:
            error = 0.0
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = max(self.output_limits[0], min(output, self.output_limits[1]))
        return output, error

def main():
    # PID controller for the pole (to keep it upright)
    pole_pid = PIDController(Kp=20.0, Ki=0.0, Kd=5.0, setpoint=0.0, output_limits=(-30.0, 30.0))
    # Proportional controller for the cart position (to keep cart at x=0)
    Kp_cart = 5.0

    dt = 1.0 / 240.0

    # Connect to PyBullet in GUI mode and set up the environment.
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(current_dir)
    cart_id = p.loadURDF("flagpole_decoupled.urdf", [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)

    # Disable default motor control.
    num_joints = p.getNumJoints(cart_id)
    for j in range(num_joints):
        p.setJointMotorControl2(
            bodyUniqueId=cart_id,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0
        )

    # Identify the pole joint (named "pole_joint").
    pole_joint_index = None
    for j in range(num_joints):
        info = p.getJointInfo(cart_id, j)
        if info[1].decode("utf-8") == "pole_joint":
            pole_joint_index = j
            break

    if pole_joint_index is None:
        print("Error: 'pole_joint' not found.")
        p.disconnect()
        return

    # Start with the pole perfectly upright.
    p.resetJointState(cart_id, pole_joint_index, 0.0)

    # Increase friction to prevent excessive sliding.
    for link_idx in range(-1, num_joints):
        p.changeDynamics(cart_id, link_idx, lateralFriction=2.0, rollingFriction=0.01, spinningFriction=0.01)

    time_data = []
    pole_error_data = []
    cart_error_data = []
    sim_time = 0.0
    max_sim_time = 10.0

    print("Closed-loop cart-pole running for 10 seconds.")
    print("Poke the rod (bob) or the cart in the GUI to disturb the system.")
    try:
        while sim_time < max_sim_time:
            start_time = time.time()

            # Read current pole angle.
            pole_angle = p.getJointState(cart_id, pole_joint_index)[0]

            # Compute PID for the pole.
            control_effort_pole, error_pole = pole_pid.compute(pole_angle, dt)

            # Get the cart base position.
            cart_pos, _ = p.getBasePositionAndOrientation(cart_id)
            error_cart = cart_pos[0]  # desired x = 0
            control_effort_cart = -Kp_cart * error_cart

            # Combine control efforts.
            total_control = control_effort_pole + control_effort_cart
            total_control = max(-30.0, min(total_control, 30.0))

            time_data.append(sim_time)
            pole_error_data.append(error_pole)
            cart_error_data.append(error_cart)
            sim_time += dt

            print(f"Time: {sim_time:.2f}s, Pole Angle: {pole_angle:.4f} rad, "
                  f"Pole Error: {error_pole:.4f}, Cart Error: {error_cart:.4f}, "
                  f"Control: {total_control:.2f}")

            # Apply torque at the pole joint.
            p.setJointMotorControl2(
                bodyUniqueId=cart_id,
                jointIndex=pole_joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=total_control
            )

            # Apply an external force to the cart base in the x-direction.
            p.applyExternalForce(cart_id, -1, [total_control, 0, 0], [0, 0, 0], p.WORLD_FRAME)

            p.stepSimulation()
            elapsed = time.time() - start_time
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        p.disconnect()

    # Plot errors.
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, pole_error_data, "r-", label="Pole Angle Error (rad)")
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time_data, cart_error_data, "b-", label="Cart Position Error (m)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.show()

if __name__ == "__main__":
    main()
