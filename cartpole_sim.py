import pybullet as p
import pybullet_data
import time
import os
import math
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-10.0, 10.0), dead_zone=1e-4):
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
    # Controller gains: tuned for moderate response when pushing or pulling.
    # For the pole angle:
    pole_pid = PIDController(Kp=10.0, Ki=0.0, Kd=2.0, setpoint=0.0, output_limits=(-10.0, 10.0))
    # For the cart's position and velocity:
    Kp_cart = 3.0  # proportional gain on cart x-position
    Kd_cart = 2.0  # damping gain on cart x-velocity

    dt = 1.0 / 240.0

    # Connect to PyBullet and set up environment.
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(current_dir)
    # Load the cart-pole URDF (flagpole.urdf)
    cart_id = p.loadURDF("flagpole.urdf", [0, 0, 0.1],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=False)

    num_joints = p.getNumJoints(cart_id)
    # Disable default motor control for all joints.
    for j in range(num_joints):
        p.setJointMotorControl2(cart_id, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    # Identify joints: pole and wheels.
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
        print("Error: Could not find all 4 wheel joints.")
        p.disconnect()
        return

    print("Wheel joints:", wheel_indices)
    print("Pole joint index:", pole_joint_index)

    # Start with the pole perfectly upright.
    p.resetJointState(cart_id, pole_joint_index, 0.0)

    # Increase friction on all links to prevent excessive sliding.
    for i in range(-1, num_joints):
        p.changeDynamics(cart_id, i, lateralFriction=12.0, rollingFriction=0.02, spinningFriction=0.02)

    # Data logging.
    time_data = []
    pole_error_data = []
    cart_error_data = []
    sim_time = 0.0
    max_sim_time = 10.0

    print("Closed-loop cart-pole running for 10 seconds.")
    print("Poke the bob (rod) or push the cart; the controller will restore equilibrium gently.")
    
    try:
        while sim_time < max_sim_time:
            start_time = time.time()

            # Read current pole angle (in radians)
            pole_angle = p.getJointState(cart_id, pole_joint_index)[0]
            # Compute PID for the pole.
            torque_pole, err_pole = pole_pid.compute(pole_angle, dt)
            
            # Get cart base position and velocity.
            cart_pos, _ = p.getBasePositionAndOrientation(cart_id)
            cart_vel, _ = p.getBaseVelocity(cart_id)
            error_cart = cart_pos[0]  # we desire x = 0
            force_cart = -Kp_cart * error_cart - Kd_cart * cart_vel[0]
            
            # Combine control efforts.
            total_control = torque_pole + force_cart
            total_control = max(-10.0, min(total_control, 10.0))  # limit control
            
            # Log data.
            time_data.append(sim_time)
            pole_error_data.append(err_pole)
            cart_error_data.append(error_cart)
            sim_time += dt
            
            print(f"Time: {sim_time:.2f}s, Pole: {pole_angle:.4f} rad, "
                  f"PoleErr: {err_pole:.4f}, CartErr: {error_cart:.4f}, "
                  f"Control: {total_control:.2f}")
            
            # Apply the computed torque to the wheel joints only.
            for wj in wheel_indices:
                p.setJointMotorControl2(cart_id, wj, p.TORQUE_CONTROL, force=total_control)
            
            # Do not apply an external force; rely only on joint torque.
            p.stepSimulation()
            elapsed = time.time() - start_time
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        p.disconnect()
    
    # Plot time-series data.
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time_data, pole_error_data, 'r-', label="Pole Angle Error (rad)")
    plt.ylabel("Pole Error (rad)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(time_data, cart_error_data, 'b-', label="Cart Position Error (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Error (m)")
    plt.legend()
    plt.grid(True)
    
    plt.suptitle("Closed-Loop Cart-Pole: Stabilization Response")
    plt.show()

if __name__ == "__main__":
    main()
