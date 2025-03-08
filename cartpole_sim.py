import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


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


class LQRController:
    def __init__(self, A, B, Q, R, output_limits=(-10.0, 10.0)):
        """
        Linear Quadratic Regulator controller.
        
        Args:
            A: System matrix
            B: Input matrix
            Q: State cost matrix
            R: Control cost matrix
            output_limits: Control output limits
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self._compute_gain()
        self.output_limits = output_limits
        
    def _compute_gain(self):
        """Compute the LQR gain matrix K."""
        # Solve the Discrete Algebraic Riccati Equation (DARE)
        # This is a simplified implementation - in practice, use scipy.linalg.solve_discrete_are
        P = np.matrix(self.Q)
        for _ in range(100):  # Iterate to convergence
            P_next = self.A.T @ P @ self.A - \
                    (self.A.T @ P @ self.B) @ np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A) + \
                    self.Q
            if np.allclose(P, P_next):
                break
            P = P_next
        
        # Compute the optimal gain K
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return K
    
    def compute(self, state_vector):
        """
        Compute control output based on current state.
        
        Args:
            state_vector: Current state [position, velocity, angle, angular_velocity]
        
        Returns:
            control: Control output
        """
        state = np.matrix(state_vector).T
        control = -float(self.K @ state)
        control = max(self.output_limits[0], min(control, self.output_limits[1]))
        return control


class MPC_SimplifiedController:
    """A simplified Model Predictive Control implementation."""
    
    def __init__(self, A, B, Q, R, N=10, output_limits=(-10.0, 10.0)):
        """
        Args:
            A: System matrix
            B: Input matrix
            Q: State cost matrix
            R: Control cost matrix
            N: Prediction horizon
            output_limits: Control output limits
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.output_limits = output_limits
        
    def compute(self, state_vector):
        """
        Compute control output based on current state using simplified MPC.
        
        In a real MPC implementation, we would solve a constrained optimization problem
        here to find the optimal control sequence over the horizon. For simplicity,
        we'll use a LQR-like approach as an approximation.
        """
        state = np.matrix(state_vector).T
        
        # Simple approach: use LQR gain for first step
        P = self.Q
        for _ in range(self.N-1, -1, -1):
            K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
            P = self.A.T @ P @ self.A - (self.A.T @ P @ self.B) @ K + self.Q
            
        control = -float(K @ state)
        control = max(self.output_limits[0], min(control, self.output_limits[1]))
        return control


class InvertedPendulumModel:
    """Mathematical model of the inverted pendulum system."""
    
    def __init__(self, M=1.0, m=0.1, L=0.5, g=9.81, b=0.1, I=0.05, air_drag=0.01):
        """
        Args:
            M: Mass of the cart (kg)
            m: Mass of the pendulum (kg)
            L: Length of the pendulum (m)
            g: Gravity acceleration (m/s^2)
            b: Friction coefficient of the cart
            I: Moment of inertia of the pendulum
            air_drag: Air drag coefficient
        """
        self.M = M
        self.m = m
        self.L = L
        self.g = g
        self.b = b
        self.I = I
        self.air_drag = air_drag
        
    def dynamics(self, t, state, u=0.0, noise=None):
        """
        Nonlinear dynamics of the inverted pendulum.
        
        Args:
            t: Time
            state: [x, x_dot, theta, theta_dot]
            u: Control input (force applied to cart)
            noise: Optional noise parameters
            
        Returns:
            derivatives: [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x, x_dot, theta, theta_dot = state
        
        # Add sensor noise if provided
        if noise is not None:
            theta += np.random.normal(0, noise.get('angle', 0))
            x += np.random.normal(0, noise.get('position', 0))
        
        # Compute nonlinear dynamics
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        
        # Include air drag on pendulum
        air_resistance = self.air_drag * theta_dot**2 * np.sign(-theta_dot)
        
        # Compute the denominator term
        d = self.I * (self.M + self.m) + self.M * self.m * self.L**2 * sin_theta**2
        
        # Compute the accelerations
        x_ddot = (u - self.b * x_dot - self.m * self.L * theta_dot**2 * sin_theta - 
                  self.m * self.L * cos_theta * (self.m * self.g * self.L * sin_theta - air_resistance) / d) / \
                 (self.M + self.m - self.m * self.L * cos_theta**2 * self.m / d)
        
        theta_ddot = (self.m * self.g * self.L * sin_theta - air_resistance - 
                      self.m * self.L * cos_theta * x_ddot) / d
        
        return [x_dot, x_ddot, theta_dot, theta_ddot]
    
    def linearize(self):
        """
        Linearize the system around the equilibrium point (upright position).
        
        Returns:
            A: System matrix
            B: Input matrix
        """
        # For small angles: sin(θ) ≈ θ, cos(θ) ≈ 1
        # The linearized system matrices
        A = np.array([
            [0, 1, 0, 0],
            [0, -self.b/(self.M+self.m), self.m*self.g*self.L/(self.M+self.m), 0],
            [0, 0, 0, 1],
            [0, -self.b*self.L/((self.M+self.m)*self.L**2 + self.I), 
             self.g*(self.M+self.m)*self.L/((self.M+self.m)*self.L**2 + self.I), 0]
        ])
        
        B = np.array([
            [0],
            [1/(self.M+self.m)],
            [0],
            [self.L/((self.M+self.m)*self.L**2 + self.I)]
        ])
        
        return A, B


def kalman_filter(z, x_prev, P_prev, F, H, Q, R):
    """
    Kalman filter implementation for state estimation with noisy measurements.
    
    Args:
        z: Measurement vector
        x_prev: Previous state estimate
        P_prev: Previous error covariance
        F: State transition matrix
        H: Measurement matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        
    Returns:
        x: Updated state estimate
        P: Updated error covariance
    """
    # Predict
    x_pred = F @ x_prev
    P_pred = F @ P_prev @ F.T + Q
    
    # Update
    y = z - H @ x_pred  # Measurement residual
    S = H @ P_pred @ H.T + R  # Residual covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
    
    x = x_pred + K @ y  # Updated state estimate
    P = (np.eye(len(x_pred)) - K @ H) @ P_pred  # Updated error covariance
    
    return x, P


def main():
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
    
    # Simulation parameters
    dt = 1.0 / 240.0
    max_sim_time = 120.0  # Extended simulation time
    
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

    # Initialize the pendulum model
    pendulum_model = InvertedPendulumModel(M=10.0, m=1.0, L=0.5, b=0.5)
    A, B = pendulum_model.linearize()
    
    # Controller setup
    # PID controller with moderate gains
    pid_controller = PIDController(Kp=15.0, Ki=0.5, Kd=5.0, setpoint=0.0, 
                                   output_limits=(-20.0, 20.0), dead_zone=0.01)
    
    # LQR controller setup
    Q = np.diag([1.0, 0.1, 10.0, 1.0])  # State cost: prioritize angle stabilization
    R = np.array([[0.1]])  # Control cost
    lqr_controller = LQRController(A, B, Q, R, output_limits=(-20.0, 20.0))
    
    # MPC controller setup
    mpc_controller = MPC_SimplifiedController(A, B, Q, R, N=15, output_limits=(-20.0, 20.0))
    
    # Initialize with a slight angle for instability
    initial_angle = 0.1  # radians
    p.resetJointState(cart_id, pole_joint_index, initial_angle)

    # Adjust dynamics for more realistic behavior
    for i in range(-1, num_joints):
        p.changeDynamics(cart_id, i, 
                         lateralFriction=1.0,
                         rollingFriction=0.01,
                         spinningFriction=0.01,
                         restitution=0.1)

    # Setup debug parameters
    # Note: PyBullet doesn't have removeUserDebugParameter, 
    # so we'll track which items to ignore during cleanup
    control_type_id = p.addUserDebugParameter("Controller (0:Off, 1:PID, 2:LQR, 3:MPC)", 0, 3, 1)
    noise_amplitude_id = p.addUserDebugParameter("Sensor Noise (0-1)", 0, 1, 0)
    filter_strength_id = p.addUserDebugParameter("Filter Strength (0-1)", 0, 1, 0.5)
    
    # Add controller tuning parameters
    pid_p_id = p.addUserDebugParameter("PID P Gain", 0, 50, 15)
    pid_i_id = p.addUserDebugParameter("PID I Gain", 0, 5, 0.5)
    pid_d_id = p.addUserDebugParameter("PID D Gain", 0, 20, 5)
    
    # Disturbance parameter
    disturbance_id = p.addUserDebugParameter("Apply Disturbance", -20, 20, 0)
    
    # Setup for real-time plotting
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    cart_positions = []
    pendulum_angles = []
    control_forces = []
    timestamps = []
    
    # State estimation setup for Kalman Filter
    # Initialize state and covariance
    x_est = np.array([[0], [0], [initial_angle], [0]])
    P_est = np.eye(4) * 0.1
    
    # Define noise covariances for Kalman filter
    Q_kalman = np.eye(4) * 0.01  # Process noise
    R_kalman = np.eye(4) * 0.1   # Measurement noise
    
    # Lines for plotting
    time_data = np.linspace(0, 10, 100)  # 10 seconds of data initially
    line_pos, = ax1.plot(time_data, np.zeros_like(time_data))
    line_angle, = ax2.plot(time_data, np.zeros_like(time_data))
    line_force, = ax3.plot(time_data, np.zeros_like(time_data))
    
    # Set up the plots
    ax1.set_ylabel('Cart Position (m)')
    ax1.set_ylim(-2, 2)
    ax1.grid(True)
    
    ax2.set_ylabel('Pendulum Angle (rad)')
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Force (N)')
    ax3.set_ylim(-20, 20)
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Text for displaying current controller and parameters
    ctrl_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes)
    
    sim_time = 0
    prev_time = time.time()
    
    # Create a debug line to visualize the force applied
    force_visual_id = p.addUserDebugLine([0, 0, 0], [0, 0, 0], [1, 0, 0], lineWidth=2)
    
    # Main simulation loop
    while sim_time < max_sim_time:
        # Get updated parameters from UI
        control_type = int(p.readUserDebugParameter(control_type_id))
        noise_amplitude = p.readUserDebugParameter(noise_amplitude_id)
        filter_strength = p.readUserDebugParameter(filter_strength_id)
        
        # Update PID parameters if changed
        pid_controller.Kp = p.readUserDebugParameter(pid_p_id)
        pid_controller.Ki = p.readUserDebugParameter(pid_i_id)
        pid_controller.Kd = p.readUserDebugParameter(pid_d_id)
        
        # Get the disturbance force
        disturbance = p.readUserDebugParameter(disturbance_id)
        
        # Get joint states
        cart_pos, cart_vel = p.getBasePositionAndOrientation(cart_id)[0][0], p.getBaseVelocity(cart_id)[0][0]
        pole_state = p.getJointState(cart_id, pole_joint_index)
        pole_angle, pole_vel = pole_state[0], pole_state[1]
        
        # Add simulated sensor noise
        if noise_amplitude > 0:
            noisy_pole_angle = pole_angle + np.random.normal(0, noise_amplitude * 0.05)
            noisy_cart_pos = cart_pos + np.random.normal(0, noise_amplitude * 0.02)
        else:
            noisy_pole_angle = pole_angle
            noisy_cart_pos = cart_pos
        
        # State vector for control
        state = np.array([cart_pos, cart_vel, pole_angle, pole_vel])
        noisy_state = np.array([noisy_cart_pos, cart_vel, noisy_pole_angle, pole_vel])
        
        # Apply Kalman filtering if filter strength > 0
        if filter_strength > 0:
            # Update measurement noise based on UI setting
            R_kalman = np.eye(4) * (0.1 + noise_amplitude * 0.2)
            
            # State transition matrix F (approximately A*dt + I)
            F = np.eye(4) + A * dt
            
            # Measurement matrix (we observe all states)
            H = np.eye(4)
            
            # Filter the noisy state
            x_est, P_est = kalman_filter(
                noisy_state.reshape(-1, 1), 
                x_est, 
                P_est, 
                F, 
                H, 
                Q_kalman, 
                R_kalman
            )
            
            # Extract the filtered state
            filtered_state = np.array([x_est[0, 0], x_est[1, 0], x_est[2, 0], x_est[3, 0]])
            
            # Blend between noisy and filtered state based on filter strength
            control_state = filtered_state * filter_strength + noisy_state * (1 - filter_strength)
        else:
            control_state = noisy_state
        
        # Compute control force based on selected controller
        force = 0
        
        if control_type == 1:  # PID
            output, error = pid_controller.compute(control_state[2], dt)  # Control based on angle
            force = output
        elif control_type == 2:  # LQR
            force = lqr_controller.compute(control_state)
        elif control_type == 3:  # MPC
            force = mpc_controller.compute(control_state)
        
        # Apply disturbance
        force += disturbance
        
        # Apply the control force to all wheels
        for wheel_idx in wheel_indices:
            p.applyExternalForce(
                cart_id, 
                wheel_idx, 
                [force, 0, 0], 
                [0, 0, 0], 
                p.LINK_FRAME
            )
        
        # Update force visualization line
        cart_position, _ = p.getBasePositionAndOrientation(cart_id)
        line_start = [cart_position[0], cart_position[1], 0.05]
        line_end = [cart_position[0] + 0.05 * force, cart_position[1], 0.05]
        p.addUserDebugLine(line_start, line_end, [1, 0, 0], lineWidth=2, replaceItemUniqueId=force_visual_id)
        
        # Store data for plotting
        cart_positions.append(cart_pos)
        pendulum_angles.append(pole_angle)
        control_forces.append(force)
        timestamps.append(sim_time)
        
        # Only keep the last 1000 points for efficiency
        if len(timestamps) > 1000:
            cart_positions.pop(0)
            pendulum_angles.pop(0)
            control_forces.pop(0)
            timestamps.pop(0)
        
        # Update plots every 100 timesteps to avoid slowing down simulation
        if len(timestamps) % 100 == 0:
            plot_time_window = 10  # seconds of data to display
            
            if len(timestamps) > 1:
                # Focus on the most recent plot_time_window seconds of data
                start_idx = 0
                if timestamps[-1] > plot_time_window:
                    for i, t in enumerate(timestamps):
                        if timestamps[-1] - t <= plot_time_window:
                            start_idx = i
                            break
                
                x_data = timestamps[start_idx:]
                x_min, x_max = x_data[0], x_data[-1]
                
                # Update lines with latest data
                line_pos.set_data(x_data, cart_positions[start_idx:])
                line_angle.set_data(x_data, pendulum_angles[start_idx:])
                line_force.set_data(x_data, control_forces[start_idx:])
                
                # Update x-axis limits to maintain a moving window
                ax1.set_xlim(x_min, x_max)
                ax2.set_xlim(x_min, x_max)
                ax3.set_xlim(x_min, x_max)
                
                # Update controller info text
                controller_names = {0: "Off", 1: "PID", 2: "LQR", 3: "MPC"}
                ctrl_text.set_text(
                    f"Controller: {controller_names[control_type]}, "
                    f"Noise: {noise_amplitude:.2f}, "
                    f"Filter: {filter_strength:.2f}"
                )
                
                # Redraw the figure
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
        
        # Step the simulation
        p.stepSimulation()
        
        # Calculate real time step and update sim_time
        current_time = time.time()
        frame_time = current_time - prev_time
        prev_time = current_time
        sim_time += frame_time
        
        # Maintain a reasonable simulation speed
        time_to_sleep = max(0, dt - frame_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        
        # Check if pendulum has fallen beyond recovery or cart is too far
            if abs(pole_angle) > 0.8 or abs(cart_pos) > 3.0:
                print("Simulation reset: pendulum fell or cart went too far")
                p.resetJointState(cart_id, pole_joint_index, 0.1)
                for wheel_idx in wheel_indices:
                    p.resetJointState(cart_id, wheel_idx, 0)
                p.resetBasePositionAndOrientation(cart_id, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
                
                # Reset controllers
                pid_controller.reset()
                
                # Clear all plots for a fresh start
                cart_positions.clear()
                pendulum_angles.clear()
                control_forces.clear()
                timestamps.clear()
                
                # Reset simulation time to make plotting restart
                sim_time = 0
    
    # Clean up resources
    p.disconnect()
    plt.ioff()
    plt.close('all')


# Function to record data for analysis
def record_experiment(controller_type, duration=20.0):
    """
    Run a controlled experiment and record data for analysis.
    
    Args:
        controller_type: 1=PID, 2=LQR, 3=MPC
        duration: Duration of recording in seconds
    
    Returns:
        Dictionary with recorded data
    """
    # Setup similar to main()
    p.connect(p.DIRECT)  # Headless mode for faster recording
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    cart_id = p.loadURDF("flagpole.urdf", [0, 0, 0.1], useFixedBase=False)
    
    # Find joints
    wheel_indices = []
    pole_joint_index = None
    for j in range(p.getNumJoints(cart_id)):
        info = p.getJointInfo(cart_id, j)
        joint_name = info[1].decode("utf-8")
        if "base_joint" in joint_name and joint_name != "pole_base_joint":
            wheel_indices.append(j)
        elif joint_name == "pole_base_joint":
            pole_joint_index = j

    # Initialize pendulum and controllers
    pendulum_model = InvertedPendulumModel(M=10.0, m=1.0, L=0.5, b=0.5)
    A, B = pendulum_model.linearize()
    
    pid_controller = PIDController(Kp=15.0, Ki=0.5, Kd=5.0, setpoint=0.0)
    Q = np.diag([1.0, 0.1, 10.0, 1.0])
    R = np.array([[0.1]])
    lqr_controller = LQRController(A, B, Q, R)
    mpc_controller = MPC_SimplifiedController(A, B, Q, R, N=15)
    
    # Initialize with a slight angle
    initial_angle = 0.1
    p.resetJointState(cart_id, pole_joint_index, initial_angle)
    
    # Data storage
    data = {
        'time': [],
        'cart_position': [],
        'pendulum_angle': [],
        'control_force': []
    }
    
    # Simulation parameters
    dt = 1.0 / 240.0
    sim_time = 0
    
    # Simulation loop
    while sim_time < duration:
        # Get states
        cart_pos, cart_vel = p.getBasePositionAndOrientation(cart_id)[0][0], p.getBaseVelocity(cart_id)[0][0]
        pole_state = p.getJointState(cart_id, pole_joint_index)
        pole_angle, pole_vel = pole_state[0], pole_state[1]
        
        state = np.array([cart_pos, cart_vel, pole_angle, pole_vel])
        
        # Compute control force
        force = 0
        if controller_type == 1:  # PID
            output, _ = pid_controller.compute(pole_angle, dt)
            force = output
        elif controller_type == 2:  # LQR
            force = lqr_controller.compute(state)
        elif controller_type == 3:  # MPC
            force = mpc_controller.compute(state)
        
        # Apply disturbance at 5 seconds
        if 5.0 <= sim_time <= 5.2:
            force += 10.0
        
        # Apply the control force
        for wheel_idx in wheel_indices:
            p.applyExternalForce(cart_id, wheel_idx, [force, 0, 0], [0, 0, 0], p.LINK_FRAME)
        
        # Record data
        data['time'].append(sim_time)
        data['cart_position'].append(cart_pos)
        data['pendulum_angle'].append(pole_angle)
        data['control_force'].append(force)
        
        # Step simulation
        p.stepSimulation()
        sim_time += dt
    
    p.disconnect()
    return data


def compare_controllers():
    """Compare and visualize the performance of different controllers."""
    # Record data for each controller
    pid_data = record_experiment(1)
    lqr_data = record_experiment(2)
    mpc_data = record_experiment(3)
    
    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Cart position
    ax1.plot(pid_data['time'], pid_data['cart_position'], 'r-', label='PID')
    ax1.plot(lqr_data['time'], lqr_data['cart_position'], 'g-', label='LQR')
    ax1.plot(mpc_data['time'], mpc_data['cart_position'], 'b-', label='MPC')
    ax1.set_ylabel('Cart Position (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Pendulum angle
    ax2.plot(pid_data['time'], pid_data['pendulum_angle'], 'r-', label='PID')
    ax2.plot(lqr_data['time'], lqr_data['pendulum_angle'], 'g-', label='LQR')
    ax2.plot(mpc_data['time'], mpc_data['pendulum_angle'], 'b-', label='MPC')
    ax2.set_ylabel('Pendulum Angle (rad)')
    ax2.grid(True)
    
    # Control force
    ax3.plot(pid_data['time'], pid_data['control_force'], 'r-', label='PID')
    ax3.plot(lqr_data['time'], lqr_data['control_force'], 'g-', label='LQR')
    ax3.plot(mpc_data['time'], mpc_data['control_force'], 'b-', label='MPC')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Force (N)')
    ax3.grid(True)
    
    # Calculate performance metrics
    def calc_metrics(data):
        # Response time (time to stabilize within 0.05 radians)
        angle_data = np.array(data['pendulum_angle'])
        time_data = np.array(data['time'])
        
        # Find where system stabilizes after disturbance at t=5s
        if len(time_data) > 0:
            disturbance_idx = np.argmin(np.abs(time_data - 5.0))
            for i in range(disturbance_idx, len(angle_data)):
                if abs(angle_data[i]) < 0.05 and all(abs(angle_data[i:i+100]) < 0.05):
                    response_time = time_data[i] - 5.0
                    break
            else:
                response_time = float('inf')
                
            # Calculate RMS error after initial settling (1-5s)
            settle_start = np.argmin(np.abs(time_data - 1.0))
            settle_end = np.argmin(np.abs(time_data - 5.0))
            rms_error = np.sqrt(np.mean(np.square(angle_data[settle_start:settle_end])))
            
            return {'response_time': response_time, 'rms_error': rms_error}
        return {'response_time': float('inf'), 'rms_error': float('inf')}
    
    pid_metrics = calc_metrics(pid_data)
    lqr_metrics = calc_metrics(lqr_data)
    mpc_metrics = calc_metrics(mpc_data)
    
    # Add metrics to plot title
    plt.suptitle(f"Controller Comparison\n" +
                f"Response Times - PID: {pid_metrics['response_time']:.3f}s, " +
                f"LQR: {lqr_metrics['response_time']:.3f}s, " +
                f"MPC: {mpc_metrics['response_time']:.3f}s\n" +
                f"RMS Errors - PID: {pid_metrics['rms_error']:.5f}, " +
                f"LQR: {lqr_metrics['rms_error']:.5f}, " +
                f"MPC: {mpc_metrics['rms_error']:.5f}")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("controller_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    # Uncomment to run controller comparison
    # compare_controllers()