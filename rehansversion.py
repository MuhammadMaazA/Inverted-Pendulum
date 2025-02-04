import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import control  # python-control library for LQR
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# ============================
# 1. Define System Parameters
# ============================
# Cart-Pendulum Parameters
m = 0.8      # Mass of pendulum (kg)
M = 10      # Mass of cart (kg)
L = 0.3      # Length of pendulum (m) - distance from pivot to pendulum mass
g = 9.81     # Gravity (m/s^2)
d = 0.1      # Damping coefficient (N·s/m) - introduced minimal damping

# Hand Properties
hand_mass = 0.1
hand_radius = 0.05
hand_start_pos = [0.8, 0, 0.05]  # Starting position (x, y, z)

# ===============================
# 2. Define State-Space Model
# ===============================
# State vector: [x, x_dot, theta, theta_dot]
A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, -(m*g)/M, 0],
    [0, 0, 0, 1],
    [0, d/(M*L), (M + m)*g/(M*L), 0]
])

B = np.array([
    [0],
    [1/M],
    [0],
    [-1/(M*L)]
])

# ===================================
# 3. Design the LQR Controller
# ===================================
# Define LQR cost matrices
Q = np.diag([100, 1, 10, 1])  # State cost weights
R = np.array([[0.1]])         # Control effort cost

# Compute the LQR gain matrix K
K, S, E = control.lqr(A, B, Q, R)
K = np.array(K).flatten()

# Scale down K to prevent excessive control forces
scaling_factor = 0.1
K_scaled = K * scaling_factor
print("Original LQR Gain K:", K)
print("Scaled LQR Gain K_scaled:", K_scaled)

# ===================================
# 4. Initialize PyBullet Environment
# ===================================
# Initialize PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load PyBullet data
p.setGravity(0, 0, -9.81)  # Set gravity

# Load ground plane
plane_id = p.loadURDF("plane.urdf")

# Define cart properties
cart_mass = M
cart_size = [0.2, 0.1, 0.05]  # Half-extents: width, depth, height
cart_start_pos = [0, 0, 0.1]   # Starting position (x, y, z)

# Define pendulum properties
pendulum_mass = m
pendulum_length = 1.2          # Increased length to make it taller (meters)
pendulum_radius = 0.02         # Thinner radius for the rod
pendulum_height = pendulum_length / 2  # Half-height for cylinder

# Create collision and visual shapes
cart_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cart_size)
cart_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=cart_size, rgbaColor=[1, 0, 0, 1])

# Pendulum as a cylinder for a round rod
pendulum_collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=pendulum_radius, height=pendulum_length)
pendulum_visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=pendulum_radius, length=pendulum_length, rgbaColor=[0, 0, 1, 1])

# Create the cart and pendulum as a single multibody system
cart_pendulum_id = p.createMultiBody(
    baseMass=cart_mass,
    baseCollisionShapeIndex=cart_collision_shape,
    baseVisualShapeIndex=cart_visual_shape,
    basePosition=cart_start_pos,
    linkMasses=[pendulum_mass],
    linkCollisionShapeIndices=[pendulum_collision_shape],
    linkVisualShapeIndices=[pendulum_visual_shape],
    linkPositions=[[0, 0, cart_size[2] + pendulum_length / 2]],  # Position pendulum on top of cart
    linkOrientations=[[0, 0, 0, 1]],  # No rotation initially
    linkInertialFramePositions=[[0, 0, 0]],
    linkInertialFrameOrientations=[[0, 0, 0, 1]],  # Identity quaternion
    linkParentIndices=[0],  # Connected to cart (base body)
    linkJointTypes=[p.JOINT_REVOLUTE],  # Hinge joint
    linkJointAxis=[[0, 0, 1]],  # Rotate around Z-axis for forward-backward swing
)

# ==============================
# 5. Define Hand for Disturbance
# ==============================
# Create collision and visual shapes for the hand
hand_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=hand_radius)
hand_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=hand_radius, rgbaColor=[0, 1, 0, 1])

# Create the hand as a separate rigid body
hand_id = p.createMultiBody(
    baseMass=hand_mass,
    baseCollisionShapeIndex=hand_collision_shape,
    baseVisualShapeIndex=hand_visual_shape,
    basePosition=hand_start_pos
)

# Disable dynamics for the hand to allow manual control
p.changeDynamics(hand_id, -1, mass=0)

# Add a user debug slider for hand movement
hand_slider = p.addUserDebugParameter("Hand Position X", -2, 2, hand_start_pos[0])

# ============================================
# 6. Disable Default Damping and Motors
# ============================================
# Reintroduce minimal damping for stability
p.changeDynamics(cart_pendulum_id, -1, linearDamping=0.1, angularDamping=0.1)
p.changeDynamics(cart_pendulum_id, 0, linearDamping=0.1, angularDamping=0.1)

# Disable default joint motor (we will control it manually)
p.setJointMotorControl2(
    bodyUniqueId=cart_pendulum_id,
    jointIndex=0,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)

# ====================================
# 7. Simulation Parameters and Logging
# ====================================
# Simulation parameters
time_step = 1. / 240.
p.setTimeStep(time_step)
p.setRealTimeSimulation(0)

# Simulation duration
sim_duration = 20  # seconds
num_steps = int(sim_duration / time_step)

# Initialize state history for plotting
state_history = np.zeros((num_steps, 4))  # [x_noisy, x_dot_noisy, theta_noisy, theta_dot_noisy]
time_history = np.zeros(num_steps)

# Initialize control force history for plotting
control_force_history = np.zeros(num_steps)

# Define maximum allowed forces to prevent runaway scenarios
max_control_force = 100.0  # Newtons
max_applied_force = 100.0  # Newtons

# ===============================
# 8. Define Sensor Noise Parameters
# ===============================
# Standard deviations for sensor noise
sigma_x = 0.01            # meters (1 cm)
sigma_x_dot = 0.1         # meters/second
sigma_theta = np.radians(0.5)    # radians (~0.5 degrees)
sigma_theta_dot = 0.05    # radians/second

# ===============================
# 9. Initialize Kalman Filter
# ===============================
# Initialize Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=4)
kf.x = np.array([0, 0, 0, 0])  # Initial state estimate
kf.P = np.eye(4) * 500          # Initial covariance
kf.F = A                         # State transition matrix
kf.H = np.eye(4)                 # Measurement function
kf.R = np.diag([sigma_x*2, sigma_x_dot**2, sigma_theta**2, sigma_theta_dot*2])  # Measurement noise
kf.Q = Q_discrete_white_noise(dim=4, dt=time_step, var=0.01)  # Process noise

# ====================================
# 10. Simulation Loop with LQR Control and Sensor Noise
# ====================================
for step in range(num_steps):
    current_time = step * time_step
    time_history[step] = current_time

    # === Disturbance via Hand ===
    # Read the hand position from the slider
    desired_hand_x = p.readUserDebugParameter(hand_slider)
    desired_hand_pos = [desired_hand_x, hand_start_pos[1], hand_start_pos[2]]

    # Get current hand position
    hand_pos, _ = p.getBasePositionAndOrientation(hand_id)

    # Calculate the change in hand position
    delta_x = desired_hand_pos[0] - hand_pos[0]

    # Limit the maximum delta_x to prevent large jumps
    max_delta_x = 0.01  # meters per timestep
    if abs(delta_x) > max_delta_x:
        delta_x = np.sign(delta_x) * max_delta_x

    # Compute the new hand position
    new_hand_pos = [
        hand_pos[0] + delta_x,
        desired_hand_pos[1],
        desired_hand_pos[2]
    ]

    # Apply the new position to the hand smoothly
    p.resetBasePositionAndOrientation(hand_id, new_hand_pos, [0, 0, 0, 1])

    # Calculate the velocity of the hand (force application)
    # Since we're using reset, approximate velocity as delta_x / time_step
    hand_velocity = delta_x / time_step

    # Define a proportional factor for the applied force based on velocity
    force_factor = 1.0  # Reduced from 10 to 1.0

    # Calculate the force to apply to the cart
    applied_force = force_factor * hand_velocity

    # Limit the applied force to prevent runaway
    applied_force = np.clip(applied_force, -max_applied_force, max_applied_force)

    # Apply the force to the cart along the X-axis
    p.applyExternalForce(
        objectUniqueId=cart_pendulum_id,
        linkIndex=-1,  # Apply to base
        forceObj=[applied_force, 0, 0],
        posObj=cart_start_pos,
        flags=p.WORLD_FRAME
    )

    # === LQR Controller ===
    # Get the cart's position and orientation
    cart_pos, cart_orn = p.getBasePositionAndOrientation(cart_pendulum_id)

    # Get the pendulum's joint state
    joint_state = p.getJointState(cart_pendulum_id, 0)
    theta = joint_state[0]          # Pendulum angle (radians)
    theta_dot = joint_state[1]      # Pendulum angular velocity (rad/s)

    # Get the cart's linear velocity
    cart_linear_vel, _ = p.getBaseVelocity(cart_pendulum_id)
    x = cart_pos[0]
    x_dot = cart_linear_vel[0]

    # === Add Sensor Noise ===
    x_noisy = x + np.random.normal(0, sigma_x)
    x_dot_noisy = x_dot + np.random.normal(0, sigma_x_dot)
    theta_noisy = theta + np.random.normal(0, sigma_theta)
    theta_dot_noisy = theta_dot + np.random.normal(0, sigma_theta_dot)

    # Current measurement vector
    z = np.array([x_noisy, x_dot_noisy, theta_noisy, theta_dot_noisy])

    # === Kalman Filter Prediction and Update ===
    kf.predict()
    kf.update(z)

    # Get the estimated state
    state_estimated = kf.x
    state_history[step, :] = state_estimated

    # Compute control input: u = -K_scaled * state_estimated
    control_force = -np.dot(K_scaled, state_estimated)

    # Limit the control force
    control_force = np.clip(control_force, -max_control_force, max_control_force)

    # Store control force in history
    control_force_history[step] = control_force

    # === Optional Visualization ===
    # Optionally, visualize the pendulum's angle using debug lines
    if step % int(1 / time_step) == 0:  # Update every second
        pendulum_base = [
            cart_pos[0],
            cart_pos[1],
            cart_pos[2] + cart_size[2]
        ]
        pendulum_end = [
            pendulum_base[0] + pendulum_length * np.sin(theta),
            pendulum_base[1],
            pendulum_base[2] - pendulum_length * np.cos(theta)
        ]
        p.addUserDebugLine(pendulum_base, pendulum_end, [0, 1, 0], 2, 0.1)

    # === Optional Logging ===
    # Uncomment the following lines to enable logging
    if step % int(1 / time_step) == 0:  # Log every second
        print(f"Time: {current_time:.2f}s, Estimated State: {state_estimated}, Control Force: {control_force:.2f} N")

    # Step the simulation
    p.stepSimulation()

    # Optional: Add a short sleep to match real-time (can be omitted for faster simulation)
    time.sleep(time_step)

# ==========================
# 11. Disconnect and Plot
# ==========================
# Disconnect from PyBullet
p.disconnect()

# ==========================
# 12. Plotting the Results
# ==========================
plt.figure(figsize=(12, 10))

# Plot Cart Position
plt.subplot(3, 1, 1)
plt.plot(time_history, state_history[:, 0], label='Estimated Cart Position (m)')
plt.title('Cart-Pendulum System Response with LQR Control, Sensor Noise, and Kalman Filter')
plt.ylabel('Cart Position (m)')
plt.legend()
plt.grid(True)

# Plot Pendulum Angle
plt.subplot(3, 1, 2)
plt.plot(time_history, np.degrees(state_history[:, 2]), label='Estimated Pendulum Angle (deg)')
plt.ylabel('Pendulum Angle (degrees)')
plt.legend()
plt.grid(True)

# Plot Control Force
plt.subplot(3, 1, 3)
plt.plot(time_history, control_force_history, label='Control Force (N)')
plt.plot(time_history, np.ones(num_steps) * max_control_force, 'r--', label='Max Control Force')
plt.plot(time_history, -np.ones(num_steps) * max_control_force, 'r--')
plt.xlabel('Time (s)')
plt.ylabel('Control Force (N)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import control  # python-control library for LQR
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# ============================
# 1. Define System Parameters
# ============================
# Cart-Pendulum Parameters
m = 0.8      # Mass of pendulum (kg)
M = 10      # Mass of cart (kg)
L = 0.3      # Length of pendulum (m) - distance from pivot to pendulum mass
g = 9.81     # Gravity (m/s^2)
d = 0.1      # Damping coefficient (N·s/m) - introduced minimal damping

# Hand Properties
hand_mass = 0.1
hand_radius = 0.05
hand_start_pos = [0.8, 0, 0.05]  # Starting position (x, y, z)

# ===============================
# 2. Define State-Space Model
# ===============================
# State vector: [x, x_dot, theta, theta_dot]
A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, -(m*g)/M, 0],
    [0, 0, 0, 1],
    [0, d/(M*L), (M + m)*g/(M*L), 0]
])

B = np.array([
    [0],
    [1/M],
    [0],
    [-1/(M*L)]
])

# ===================================
# 3. Design the LQR Controller
# ===================================
# Define LQR cost matrices
Q = np.diag([100, 1, 10, 1])  # State cost weights
R = np.array([[0.1]])         # Control effort cost

# Compute the LQR gain matrix K
K, S, E = control.lqr(A, B, Q, R)
K = np.array(K).flatten()

# Scale down K to prevent excessive control forces
scaling_factor = 0.1
K_scaled = K * scaling_factor
print("Original LQR Gain K:", K)
print("Scaled LQR Gain K_scaled:", K_scaled)

# ===================================
# 4. Initialize PyBullet Environment
# ===================================
# Initialize PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load PyBullet data
p.setGravity(0, 0, -9.81)  # Set gravity

# Load ground plane
plane_id = p.loadURDF("plane.urdf")

# Define cart properties
cart_mass = M
cart_size = [0.2, 0.1, 0.05]  # Half-extents: width, depth, height
cart_start_pos = [0, 0, 0.1]   # Starting position (x, y, z)

# Define pendulum properties
pendulum_mass = m
pendulum_length = 1.2          # Increased length to make it taller (meters)
pendulum_radius = 0.02         # Thinner radius for the rod
pendulum_height = pendulum_length / 2  # Half-height for cylinder

# Create collision and visual shapes
cart_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cart_size)
cart_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=cart_size, rgbaColor=[1, 0, 0, 1])

# Pendulum as a cylinder for a round rod
pendulum_collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=pendulum_radius, height=pendulum_length)
pendulum_visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=pendulum_radius, length=pendulum_length, rgbaColor=[0, 0, 1, 1])

# Create the cart and pendulum as a single multibody system
cart_pendulum_id = p.createMultiBody(
    baseMass=cart_mass,
    baseCollisionShapeIndex=cart_collision_shape,
    baseVisualShapeIndex=cart_visual_shape,
    basePosition=cart_start_pos,
    linkMasses=[pendulum_mass],
    linkCollisionShapeIndices=[pendulum_collision_shape],
    linkVisualShapeIndices=[pendulum_visual_shape],
    linkPositions=[[0, 0, cart_size[2] + pendulum_length / 2]],  # Position pendulum on top of cart
    linkOrientations=[[0, 0, 0, 1]],  # No rotation initially
    linkInertialFramePositions=[[0, 0, 0]],
    linkInertialFrameOrientations=[[0, 0, 0, 1]],  # Identity quaternion
    linkParentIndices=[0],  # Connected to cart (base body)
    linkJointTypes=[p.JOINT_REVOLUTE],  # Hinge joint
    linkJointAxis=[[0, 0, 1]],  # Rotate around Z-axis for forward-backward swing
)

# ==============================
# 5. Define Hand for Disturbance
# ==============================
# Create collision and visual shapes for the hand
hand_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=hand_radius)
hand_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=hand_radius, rgbaColor=[0, 1, 0, 1])

# Create the hand as a separate rigid body
hand_id = p.createMultiBody(
    baseMass=hand_mass,
    baseCollisionShapeIndex=hand_collision_shape,
    baseVisualShapeIndex=hand_visual_shape,
    basePosition=hand_start_pos
)

# Disable dynamics for the hand to allow manual control
p.changeDynamics(hand_id, -1, mass=0)

# Add a user debug slider for hand movement
hand_slider = p.addUserDebugParameter("Hand Position X", -2, 2, hand_start_pos[0])

# ============================================
# 6. Disable Default Damping and Motors
# ============================================
# Reintroduce minimal damping for stability
p.changeDynamics(cart_pendulum_id, -1, linearDamping=0.1, angularDamping=0.1)
p.changeDynamics(cart_pendulum_id, 0, linearDamping=0.1, angularDamping=0.1)

# Disable default joint motor (we will control it manually)
p.setJointMotorControl2(
    bodyUniqueId=cart_pendulum_id,
    jointIndex=0,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)

# ====================================
# 7. Simulation Parameters and Logging
# ====================================
# Simulation parameters
time_step = 1. / 240.
p.setTimeStep(time_step)
p.setRealTimeSimulation(0)

# Simulation duration
sim_duration = 20  # seconds
num_steps = int(sim_duration / time_step)

# Initialize state history for plotting
state_history = np.zeros((num_steps, 4))  # [x_noisy, x_dot_noisy, theta_noisy, theta_dot_noisy]
time_history = np.zeros(num_steps)

# Initialize control force history for plotting
control_force_history = np.zeros(num_steps)

# Define maximum allowed forces to prevent runaway scenarios
max_control_force = 100.0  # Newtons
max_applied_force = 100.0  # Newtons

# ===============================
# 8. Define Sensor Noise Parameters
# ===============================
# Standard deviations for sensor noise
sigma_x = 0.01            # meters (1 cm)
sigma_x_dot = 0.1         # meters/second
sigma_theta = np.radians(0.5)    # radians (~0.5 degrees)
sigma_theta_dot = 0.05    # radians/second

# ===============================
# 9. Initialize Kalman Filter
# ===============================
# Initialize Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=4)
kf.x = np.array([0, 0, 0, 0])  # Initial state estimate
kf.P = np.eye(4) * 500          # Initial covariance
kf.F = A                         # State transition matrix
kf.H = np.eye(4)                 # Measurement function
kf.R = np.diag([sigma_x*2, sigma_x_dot2, sigma_theta2, sigma_theta_dot*2])  # Measurement noise
kf.Q = Q_discrete_white_noise(dim=4, dt=time_step, var=0.01)  # Process noise

# ====================================
# 10. Simulation Loop with LQR Control and Sensor Noise
# ====================================
for step in range(num_steps):
    current_time = step * time_step
    time_history[step] = current_time

    # === Disturbance via Hand ===
    # Read the hand position from the slider
    desired_hand_x = p.readUserDebugParameter(hand_slider)
    desired_hand_pos = [desired_hand_x, hand_start_pos[1], hand_start_pos[2]]

    # Get current hand position
    hand_pos, _ = p.getBasePositionAndOrientation(hand_id)

    # Calculate the change in hand position
    delta_x = desired_hand_pos[0] - hand_pos[0]

    # Limit the maximum delta_x to prevent large jumps
    max_delta_x = 0.01  # meters per timestep
    if abs(delta_x) > max_delta_x:
        delta_x = np.sign(delta_x) * max_delta_x

    # Compute the new hand position
    new_hand_pos = [
        hand_pos[0] + delta_x,
        desired_hand_pos[1],
        desired_hand_pos[2]
    ]

    # Apply the new position to the hand smoothly
    p.resetBasePositionAndOrientation(hand_id, new_hand_pos, [0, 0, 0, 1])

    # Calculate the velocity of the hand (force application)
    # Since we're using reset, approximate velocity as delta_x / time_step
    hand_velocity = delta_x / time_step

    # Define a proportional factor for the applied force based on velocity
    force_factor = 1.0  # Reduced from 10 to 1.0

    # Calculate the force to apply to the cart
    applied_force = force_factor * hand_velocity

    # Limit the applied force to prevent runaway
    applied_force = np.clip(applied_force, -max_applied_force, max_applied_force)

    # Apply the force to the cart along the X-axis
    p.applyExternalForce(
        objectUniqueId=cart_pendulum_id,
        linkIndex=-1,  # Apply to base
        forceObj=[applied_force, 0, 0],
        posObj=cart_start_pos,
        flags=p.WORLD_FRAME
    )

    # === LQR Controller ===
    # Get the cart's position and orientation
    cart_pos, cart_orn = p.getBasePositionAndOrientation(cart_pendulum_id)

    # Get the pendulum's joint state
    joint_state = p.getJointState(cart_pendulum_id, 0)
    theta = joint_state[0]          # Pendulum angle (radians)
    theta_dot = joint_state[1]      # Pendulum angular velocity (rad/s)

    # Get the cart's linear velocity
    cart_linear_vel, _ = p.getBaseVelocity(cart_pendulum_id)
    x = cart_pos[0]
    x_dot = cart_linear_vel[0]

    # === Add Sensor Noise ===
    x_noisy = x + np.random.normal(0, sigma_x)
    x_dot_noisy = x_dot + np.random.normal(0, sigma_x_dot)
    theta_noisy = theta + np.random.normal(0, sigma_theta)
    theta_dot_noisy = theta_dot + np.random.normal(0, sigma_theta_dot)

    # Current measurement vector
    z = np.array([x_noisy, x_dot_noisy, theta_noisy, theta_dot_noisy])

    # === Kalman Filter Prediction and Update ===
    kf.predict()
    kf.update(z)

    # Get the estimated state
    state_estimated = kf.x
    state_history[step, :] = state_estimated

    # Compute control input: u = -K_scaled * state_estimated
    control_force = -np.dot(K_scaled, state_estimated)

    # Limit the control force
    control_force = np.clip(control_force, -max_control_force, max_control_force)

    # Store control force in history
    control_force_history[step] = control_force

    # === Optional Visualization ===
    # Optionally, visualize the pendulum's angle using debug lines
    if step % int(1 / time_step) == 0:  # Update every second
        pendulum_base = [
            cart_pos[0],
            cart_pos[1],
            cart_pos[2] + cart_size[2]
        ]
        pendulum_end = [
            pendulum_base[0] + pendulum_length * np.sin(theta),
            pendulum_base[1],
            pendulum_base[2] - pendulum_length * np.cos(theta)
        ]
        p.addUserDebugLine(pendulum_base, pendulum_end, [0, 1, 0], 2, 0.1)

    # === Optional Logging ===
    # Uncomment the following lines to enable logging
    if step % int(1 / time_step) == 0:  # Log every second
        print(f"Time: {current_time:.2f}s, Estimated State: {state_estimated}, Control Force: {control_force:.2f} N")

    # Step the simulation
    p.stepSimulation()

    # Optional: Add a short sleep to match real-time (can be omitted for faster simulation)
    time.sleep(time_step)

# ==========================
# 11. Disconnect and Plot
# ==========================
# Disconnect from PyBullet
p.disconnect()

# ==========================
# 12. Plotting the Results
# ==========================
plt.figure(figsize=(12, 10))

# Plot Cart Position
plt.subplot(3, 1, 1)
plt.plot(time_history, state_history[:, 0], label='Estimated Cart Position (m)')
plt.title('Cart-Pendulum System Response with LQR Control, Sensor Noise, and Kalman Filter')
plt.ylabel('Cart Position (m)')
plt.legend()
plt.grid(True)

# Plot Pendulum Angle
plt.subplot(3, 1, 2)
plt.plot(time_history, np.degrees(state_history[:, 2]), label='Estimated Pendulum Angle (deg)')
plt.ylabel('Pendulum Angle (degrees)')
plt.legend()
plt.grid(True)

# Plot Control Force
plt.subplot(3, 1, 3)
plt.plot(time_history, control_force_history, label='Control Force (N)')
plt.plot(time_history, np.ones(num_steps) * max_control_force, 'r--', label='Max Control Force')
plt.plot(time_history, -np.ones(num_steps) * max_control_force, 'r--')
plt.xlabel('Time (s)')
plt.ylabel('Control Force (N)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()