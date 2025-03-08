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
    pi