import pybullet as p
import pybullet_data
import time
import os
import math
import matplotlib.pyplot as plt
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-10.0, 10.0), dead_zone=1e-4):
        """
        PID Controller with dead zone option.
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
        """Compute control output based on measurement."""
        error = measurement - self.setpoint
        if abs(error) < self.dead_zone:
            error = 0.0
        
        # Anti-windup: limit integral term
        if self.Ki != 0:
            self.integral = max(self.output_limits[0]/self.Ki, 
                               min(self.integral + error * dt, 
                                   self.output_limits[1]/self.Ki))
        else:
            self.integral += error * dt
            
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = error
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = max(self.output_limits[0], min(output, self.output_limits[1]))
        
        return output, error

    def reset(self):
        """Reset the controller's internal state."""
        self.integral = 0.0
        self.previous_error = 0.0

def main():
    # Controller parameters - reduced gains to allow more natural motion
    # For the pole angle (more gentle control):
    pole_pid = PIDController(Kp=5.0, Ki=0.1, Kd=1.0, setpoint=0.0, 
                            output_limits=(-8.0, 8.0), dead_zone=0.01)
    
    # For cart position - REMOVED to allow manual positioning
    # Instead, we'll use damping to prevent excessive motion
    cart_damping = 1.5  # Damping coefficient for cart motion
    
    # Simulation parameters
    dt = 1.0 / 240.0
    max_sim_time = 100.0

    # Connect to PyBullet and set up environment
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Load the cart-pole URDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(current_dir)
    cart_id = p.loadURDF("flagpole.urdf", [0, 0, 0.1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       useFixedBase=False)

    # Set up PyBullet parameters for better interaction
    p.setPhysicsEngineParameter(enableFileCaching=0,
                              numSolverIterations=50,
                              numSubSteps=4)
    
    # Add debug parameters for real-time tuning
    param_ids = []
    pole_kp_id = p.addUserDebugParameter("Pole Kp", 0, 20, 5.0)
    pole_ki_id = p.addUserDebugParameter("Pole Ki", 0, 1, 0.1)
    pole_kd_id = p.addUserDebugParameter("Pole Kd", 0, 5, 1.0)
    damping_id = p.addUserDebugParameter("Cart Damping", 0, 5, 1.5)
    param_ids.extend([pole_kp_id, pole_ki_id, pole_kd_id, damping_id])

    # Disable default motor control and get joint information
    num_joints = p.getNumJoints(cart_id)
    for j in range(num_joints):
        p.setJointMotorControl2(cart_id, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    # Identify joints: pole and wheels
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

    # Initialize the pole with a slight offset to make it unstable
    p.resetJointState(cart_id, pole_joint_index, 0.1)  # Slight initial angle

    # Adjust dynamics for more realistic behavior
    for i in range(-1, num_joints):
        p.changeDynamics(cart_id, i, 
                       lateralFriction=1.0,       # Reduced to allow more natural slide
                       rollingFriction=0.01,      # Reduced for smoother movement
                       spinningFriction=0.01,     # Reduced for smoother movement
                       restitution=0.1)           # Some bounce on collision

    # Data logging
    time_data = []
    pole_angle_data = []
    cart_position_data = []
    control_force_data = []
    sim_time = 0.0
    
    # Variables for user interaction
    force_scaling = 5.0     # Scale applied mouse drag forces
    last_draginfo = None
    
    # Control mode state - start with control enabled, can be toggled
    control_enabled = True
    control_text_id = p.addUserDebugText("Control: ENABLED", [0, 0, 0.5], [0, 1, 0])
    
    # Add toggle button for control
    toggle_button = p.addUserDebugParameter("Toggle Control", 1, 0, 1)
    last_toggle_state = p.readUserDebugParameter(toggle_button)
    
    print("\nInverted Pendulum Simulation")
    print("----------------------------")
    print("- Click and drag the cart to move it manually")
    print("- The controller will try to keep the pole upright")
    print("- Toggle the control system on/off with the control button")
    print("- Adjust PID parameters in real-time with the sliders")
    print("- Press Ctrl+C to exit\n")

    try:
        while sim_time < max_sim_time:
            start_time = time.time()
            
            # Check if control toggle button was pressed
            current_toggle = p.readUserDebugParameter(toggle_button)
            if abs(current_toggle - last_toggle_state) > 0.5:
                control_enabled = not control_enabled
                last_toggle_state = current_toggle
                
                # Update display text
                p.removeUserDebugItem(control_text_id)
                if control_enabled:
                    control_text_id = p.addUserDebugText("Control: ENABLED", [0, 0, 0.5], [0, 1, 0])
                    pole_pid.reset()  # Reset controller state when re-enabling
                else:
                    control_text_id = p.addUserDebugText("Control: DISABLED", [0, 0, 0.5], [1, 0, 0])
            
            # Update PID parameters from sliders
            pole_pid.Kp = p.readUserDebugParameter(pole_kp_id)
            pole_pid.Ki = p.readUserDebugParameter(pole_ki_id)
            pole_pid.Kd = p.readUserDebugParameter(pole_kd_id)
            cart_damping = p.readUserDebugParameter(damping_id)
            
            # Handle mouse interaction for dragging the cart
            mouse_events = p.getMouseEvents()
            for e in mouse_events:
                if e[0] == 2:  # Mouse move with button down
                    if e[3] == 0 and e[4] == cart_id:  # Left button & hit our cart
                        rayFrom, rayTo, rayInfo = p.getDebugVisualizerCamera()[10:13]
                        camPos = np.array(rayFrom)
                        rayFwd = np.array(rayTo) - np.array(rayFrom)
                        rayLen = np.linalg.norm(rayFwd)
                        rayNorm = rayFwd / rayLen
                        
                        # Get hit position
                        hitPos = camPos + rayNorm * rayInfo[3]
                        # Apply force in the direction of movement
                        if last_draginfo is not None:
                            drag_direction = np.array(hitPos) - np.array(last_draginfo)
                            # Only apply force in x direction (2D constraint)
                            if abs(drag_direction[0]) > 0.001:  # Small threshold to avoid noise
                                p.applyExternalForce(cart_id, -1, 
                                                   [drag_direction[0] * force_scaling, 0, 0], 
                                                   hitPos, p.WORLD_FRAME)
                        last_draginfo = hitPos
                elif e[0] == 5:  # Button up
                    last_draginfo = None
            
            # Read current pole angle and cart position
            pole_angle = p.getJointState(cart_id, pole_joint_index)[0]
            cart_pos, cart_orient = p.getBasePositionAndOrientation(cart_id)
            cart_vel, ang_vel = p.getBaseVelocity(cart_id)
            
            # Calculate control action
            total_control = 0
            if control_enabled:
                # Compute PID for pole angle
                pole_torque, pole_error = pole_pid.compute(pole_angle, dt)
                
                # Apply damping to cart motion (instead of position control)
                # This allows manual positioning while preventing excessive movement
                damping_force = -cart_damping * cart_vel[0]
                
                # Combine control efforts
                total_control = pole_torque + damping_force
                total_control = max(-8.0, min(total_control, 8.0))  # Limit control force
            
            # Apply the computed force to wheel joints for controlled movement
            for wj in wheel_indices:
                p.setJointMotorControl2(cart_id, wj, p.TORQUE_CONTROL, force=total_control)
            
            # Log data
            time_data.append(sim_time)
            pole_angle_data.append(pole_angle)
            cart_position_data.append(cart_pos[0])
            control_force_data.append(total_control)
            
            # Print status at reduced frequency
            if int(sim_time * 10) % 5 == 0:
                mode = "ENABLED" if control_enabled else "DISABLED"
                print(f"Time: {sim_time:.1f}s | Control: {mode} | "
                      f"Pole: {pole_angle:.3f} rad | Cart: {cart_pos[0]:.3f} m | "
                      f"Force: {total_control:.2f}")
            
            # Advance simulation
            p.stepSimulation()
            sim_time += dt
            
            # Maintain real-time factor
            elapsed = time.time() - start_time
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        # Clean up debug items
        for id in param_ids:
            p.removeUserDebugParameter(id)
        p.removeUserDebugItem(control_text_id)
        p.removeUserDebugParameter(toggle_button)
        
        # Plot results if we have enough data
        if len(time_data) > 10:
            plt.figure(figsize=(12, 9))
            
            plt.subplot(3, 1, 1)
            plt.plot(time_data, pole_angle_data, 'r-', label="Pole Angle (rad)")
            plt.ylabel("Angle (rad)")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.plot(time_data, cart_position_data, 'b-', label="Cart Position (m)")
            plt.ylabel("Position (m)")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(time_data, control_force_data, 'g-', label="Control Force (N)")
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.legend()
            plt.grid(True)
            
            plt.suptitle("Inverted Pendulum Simulation Results")
            plt.tight_layout()
            plt.show()
        
        p.disconnect()
        print("Simulation ended.")

if __name__ == "__main__":
    main()