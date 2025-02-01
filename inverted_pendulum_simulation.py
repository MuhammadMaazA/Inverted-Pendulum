"""
inverted_pendulum_simulation.py

A complete simulation of an inverted pendulum on a cart with:
  - Nonlinear dynamics (with optional air drag and pivot damping).
  - Sensor noise injection and simple low-pass filtering.
  - Three controllers: PID, pole placement (LQR style), and a nonlinear energy-based controller.
  - Two disturbance modes: external force disturbance and a tip disturbance.
  - Real-time animation and interactive plots (with sliders for tuning).

Dependencies:
    numpy, matplotlib, scipy

To install dependencies in your active environment, run:
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, RadioButtons
from scipy.integrate import solve_ivp
import time

##########################################
# 1. SYSTEM PARAMETERS & SETTINGS
##########################################

# Physical parameters
params = {
    'M': 0.5,       # Mass of cart (kg)
    'm': 0.05,      # Mass of pendulum (kg) [50 grams]
    'L': 0.45,      # Length of rod (m) [choose between 0.4-0.5 m]
    'g': 9.81,      # Gravity (m/s^2)
    'b_cart': 0.1,  # Damping coefficient on cart (N·s/m)
    'b_pend': 0.005 # Damping at pivot (N·m·s)
}

# Simulation time settings
sim_settings = {
    't_start': 0,
    't_end': 10.0,      # Total simulation time in seconds
    'dt': 0.005         # Integration time step (s)
}

# Sensor noise standard deviations (for simulation)
noise_params = {
    'x_noise': 0.005,        # noise on cart position (m)
    'xdot_noise': 0.005,     # noise on cart velocity (m/s)
    'theta_noise': 0.002,    # noise on pendulum angle (rad)
    'thetadot_noise': 0.002  # noise on pendulum angular velocity (rad/s)
}

# Disturbance settings:
# At a specified time, apply an external force on the cart.
disturbance = {
    'force_start': 2.0,    # time to start disturbance (s)
    'force_duration': 0.1, # duration of disturbance (s)
    'force_magnitude': 10.0, # force magnitude (N)
    # Additionally, a tip disturbance to the pendulum angle
    'tip_disturbance_time': 4.0,  # time when the pendulum tip is disturbed
    'tip_angle_offset': np.deg2rad(10)  # add 10 degrees (in radians) offset
}

##########################################
# 2. CONTROLLER IMPLEMENTATIONS
##########################################

# --- 2a. PID Controller ---
class PIDController:
    def __init__(self, kp=100.0, ki=0.0, kd=20.0):
        # PID gains (for combined stabilization)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0

    def reset(self):
        self.integral = 0
        self.last_error = 0

    def compute(self, t, state, state_filtered, ref):
        """
        state: measured state vector [x, xdot, theta, thetadot]
        ref: reference vector [x_ref, theta_ref] (theta_ref = 0 for upright)
        We'll combine errors for cart and pendulum.
        """
        # Use filtered measurements for control
        x, xdot, theta, thetadot = state_filtered

        # Error definitions:
        error_theta = ref[1] - theta  # For pendulum (target 0 radians)
        error_x = ref[0] - x          # For cart (could be 0 or a trajectory)

        # For this example, we combine the errors into one control input:
        # (The controller cannot act on the pivot so we apply a force on the cart)
        error = error_theta + 0.2 * error_x  # weight the cart error lower

        # Integral and derivative (for theta error)
        self.integral += error * sim_settings['dt']
        derivative = (error - self.last_error) / sim_settings['dt']
        self.last_error = error

        F = self.kp * error + self.ki * self.integral + self.kd * derivative

        return F

# --- 2b. Pole Placement / LQR Controller ---
class PolePlacementController:
    def __init__(self):
        # Gains from an LQR design on the linearized model.
        # For our parameters and linearization around [0,0,0,0], these gains are example values.
        # (They may be tuned further.)
        # K is a row vector for state feedback: F = -K [x, xdot, theta, thetadot]^T
        self.K = np.array([ -80.0, -15.0, 150.0, 25.0 ])

    def compute(self, t, state, state_filtered, ref):
        """
        Use full state feedback.
        ref: reference for cart position (for pendulum, reference is 0)
        """
        x, xdot, theta, thetadot = state_filtered
        # Create error state relative to desired equilibrium:
        # For cart, we want to track ref[0] (which may be nonzero in moving mode).
        # For pendulum, the desired angle is always 0.
        x_error = x - ref[0]
        theta_error = theta - 0.0  # desired pendulum angle is 0
        error_state = np.array([ x_error, xdot, theta_error, thetadot ])
        # Compute force
        F = -np.dot(self.K, error_state)
        return F

# --- 2c. Nonlinear (Energy-based) Controller ---
class NonlinearController:
    def __init__(self):
        # Gains for energy shaping and stabilization.
        self.k_energy = 5.0      # energy gain
        self.k_stabilize = 50.0  # stabilization gain (for switching when near upright)

    def compute(self, t, state, state_filtered, ref):
        """
        This controller applies a swing-up control law based on energy shaping.
        When the pendulum is far from the upright (|theta| > threshold),
        it injects energy to drive the pendulum upward.
        When near upright, it switches to a linear stabilization (like a PD).
        """
        x, xdot, theta, thetadot = state_filtered

        # Define a threshold for switching to stabilization mode (in radians)
        threshold = np.deg2rad(12)  # 12 degrees

        # Total energy of the pendulum (assuming point mass at end):
        # Kinetic Energy (rotational): 0.5 * m * (L*thetadot)^2
        # Potential Energy: m*g*L*(1 - cos(theta))
        m = params['m']
        L = params['L']
        g = params['g']
        E = 0.5 * m * (L * thetadot)**2 + m * g * L * (1 - np.cos(theta))
        # Desired energy at the upright position is defined here as:
        E_des = m * g * L

        if np.abs(theta) > threshold:
            # Swing-up control: drive the energy error toward zero.
            energy_error = E - E_des
            # Control law: apply force in the direction of thetadot*cos(theta)
            F = self.k_energy * energy_error * np.sign(thetadot * np.cos(theta))
        else:
            # When near upright, switch to a PD-like stabilization on theta.
            error = 0.0 - theta
            derror = 0.0 - thetadot
            F = self.k_stabilize * (error + 0.5 * derror)
        # Optionally add a small component to control cart position
        F += -5.0 * x - 2.0 * xdot
        return F

##########################################
# 3. SENSOR NOISE AND FILTERING
##########################################

class Sensor:
    def __init__(self, noise_params):
        self.noise_params = noise_params
        # For filtering, we use a simple first-order low-pass filter.
        self.filtered = None
        self.alpha = 0.95  # filter coefficient (0 < alpha < 1); higher means smoother

    def measure(self, true_state):
        """
        Add Gaussian noise to the true state.
        State: [x, xdot, theta, thetadot]
        """
        noise = np.array([
            np.random.normal(0, self.noise_params['x_noise']),
            np.random.normal(0, self.noise_params['xdot_noise']),
            np.random.normal(0, self.noise_params['theta_noise']),
            np.random.normal(0, self.noise_params['thetadot_noise'])
        ])
        measured = true_state + noise
        return measured

    def low_pass_filter(self, measurement):
        """
        Simple low-pass filter to smooth the noisy measurement.
        """
        if self.filtered is None:
            self.filtered = measurement
        else:
            self.filtered = self.alpha * self.filtered + (1 - self.alpha) * measurement
        return self.filtered

##########################################
# 4. DYNAMICS OF THE SYSTEM
##########################################

def dynamics(t, state, controller, sensor, params, disturbance, ref, mode):
    """
    Compute the state derivatives for the inverted pendulum on a cart.
    State vector: [x, xdot, theta, thetadot]
    
    mode: string that selects simulation mode:
          'disturbance' -> apply an external force disturbance (and tip disturbance)
          'moving'      -> track a cart position reference (e.g., a slow sine wave)
    """
    M = params['M']
    m = params['m']
    L = params['L']
    g = params['g']
    b_cart = params['b_cart']
    b_pend = params['b_pend']

    x, xdot, theta, thetadot = state

    # Simulate sensor measurement and filtering.
    measurement = sensor.measure(state)
    state_filtered = sensor.low_pass_filter(measurement)

    # Reference update for moving mode (for cart position tracking)
    if mode == 'moving':
        # For example, a slow sine trajectory for the cart.
        x_ref = 0.2 * np.sin(0.5 * t)
    else:
        x_ref = 0.0
    ref_updated = [x_ref, 0.0]  # we always want pendulum to be 0 (upright)

    # Compute control force from chosen controller (force applied to cart)
    F_control = controller.compute(t, state, state_filtered, ref_updated)

    # Apply external disturbance force if within disturbance window (only in disturbance mode)
    if mode == 'disturbance':
        if disturbance['force_start'] <= t < (disturbance['force_start'] + disturbance['force_duration']):
            F_control += disturbance['force_magnitude']
        # Apply a tip disturbance (simulate an impulse to pendulum angle)
        if np.isclose(t, disturbance['tip_disturbance_time'], atol=sim_settings['dt']):
            theta += disturbance['tip_angle_offset']

    # Save the control force for logging purposes by storing it as an attribute.
    dynamics.F_control = F_control

    # Nonlinear dynamics (derived from Newton's laws)
    denom = M + m * (1 - np.cos(theta)**2)
    xddot = (F_control + m * np.sin(theta) * (L * thetadot**2 + g * np.cos(theta)) - b_cart * xdot) / denom
    thetaddot = (-F_control * np.cos(theta) - m * L * thetadot**2 * np.sin(theta) * np.cos(theta)
                 - (M + m) * g * np.sin(theta) - b_pend * thetadot) / (L * denom)

    return [xdot, xddot, thetadot, thetaddot]

##########################################
# 5. SIMULATION FUNCTION
##########################################

def run_simulation(controller_type='PID', mode='disturbance'):
    """
    Run the simulation using one of the three controllers:
        controller_type: 'PID', 'Pole', or 'Nonlinear'
    mode: 'disturbance' or 'moving'
    Returns: time array, state history, control force history.
    """
    # Choose controller instance based on input
    if controller_type == 'PID':
        controller = PIDController(kp=100.0, ki=0.0, kd=20.0)
    elif controller_type == 'Pole':
        controller = PolePlacementController()
    elif controller_type == 'Nonlinear':
        controller = NonlinearController()
    else:
        raise ValueError("Unknown controller type")

    # Create sensor instance
    sensor = Sensor(noise_params)

    # Define time span and evaluation points
    t_span = (sim_settings['t_start'], sim_settings['t_end'])
    t_eval = np.arange(sim_settings['t_start'], sim_settings['t_end'], sim_settings['dt'])

    # For logging control force at each time step
    global control_force_log
    control_force_log = []

    # Define a wrapper dynamics function that logs the control force.
    def dynamics_wrapper(t, state):
        derivs = dynamics(t, state, controller, sensor, params, disturbance, ref=[0,0], mode=mode)
        control_force_log.append(dynamics.F_control)
        return derivs

    # Initial state: [x, xdot, theta, thetadot]
    state0 = [0.0, 0.0, np.deg2rad(2.0), 0.0]

    # Solve the ODE
    sol = solve_ivp(dynamics_wrapper, t_span, state0, t_eval=t_eval, method='RK45')
    return sol.t, sol.y, np.array(control_force_log)

##########################################
# 6. VISUALIZATION AND ANIMATION
##########################################

def animate_simulation(t, state_history, control_force_history, controller_type, mode):
    """
    Animate the cart and pendulum motion and plot time series of x, theta, and control force.
    """
    # Extract states
    x = state_history[0, :]
    theta = state_history[2, :]  # in radians

    # Set up the figure with two subplots:
    # Left: animation of cart and pendulum.
    # Right: plots of x, theta (in degrees), and control force vs. time.
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2)
    ax_anim = fig.add_subplot(gs[:, 0])
    ax_plots = fig.add_subplot(gs[0, 1])
    ax_force = fig.add_subplot(gs[1, 1])

    # Animation axis settings
    ax_anim.set_xlim(-1, 1)
    ax_anim.set_ylim(-0.2, 1.2)
    ax_anim.set_aspect('equal')
    ax_anim.set_title(f'Inverted Pendulum Simulation\nController: {controller_type}, Mode: {mode}')
    ax_anim.set_xlabel('Cart Position (m)')
    ax_anim.set_ylabel('Height (m)')

    # Elements to animate:
    cart_width = 0.2
    cart_height = 0.1
    rod_length = params['L']

    # Cart represented as a rectangle patch and pendulum as a line.
    cart_patch = plt.Rectangle((x[0]-cart_width/2, 0), cart_width, cart_height, fc='blue', ec='black')
    ax_anim.add_patch(cart_patch)
    # Pendulum line: from the top of the cart to the pendulum bob.
    line, = ax_anim.plot([], [], lw=3, c='red')
    bob, = ax_anim.plot([], [], 'o', c='black', markersize=8)

    # Set up time-series plots for cart position and control force.
    l1, = ax_plots.plot([], [], 'b-', label='Cart Position (m)')
    ax_plots.set_xlim(t[0], t[-1])
    ax_plots.set_ylim(np.min(x)-0.1, np.max(x)+0.1)
    ax_plots.set_xlabel('Time (s)')
    ax_plots.set_ylabel('x (m)')
    ax_plots.legend()
    ax_plots.grid(True)

    l2, = ax_force.plot([], [], 'r-', label='Control Force (N)')
    ax_force.set_xlim(t[0], t[-1])
    ax_force.set_ylim(np.min(control_force_history)-1, np.max(control_force_history)+1)
    ax_force.set_xlabel('Time (s)')
    ax_force.set_ylabel('Force (N)')
    ax_force.legend()
    ax_force.grid(True)

    # Data containers for plots
    x_data, force_data, time_data = [], [], []

    # Animation function
    def animate(i):
        # Update cart and pendulum position
        cart_x = x[i]
        cart_patch.set_xy((cart_x - cart_width/2, 0))
        # The pendulum pivot is at the top center of the cart.
        pivot = (cart_x, cart_height)
        bob_x = pivot[0] + rod_length * np.sin(theta[i])
        bob_y = pivot[1] + rod_length * np.cos(theta[i])
        line.set_data([pivot[0], bob_x], [pivot[1], bob_y])
        bob.set_data([bob_x], [bob_y])  # Fixed: pass as sequences

        # Update time-series plots
        time_data.append(t[i])
        x_data.append(cart_x)
        force_data.append(control_force_history[i])
        l1.set_data(time_data, x_data)
        l2.set_data(time_data, force_data)
        return cart_patch, line, bob, l1, l2

    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=sim_settings['dt']*1000, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()

##########################################
# 7. INTERACTIVE PARAMETER ADJUSTMENT (Optional)
##########################################

def interactive_simulation():
    """
    Create an interactive simulation using matplotlib sliders and radio buttons.
    Adjust controller type and key gains in real time and re-run simulation.
    """
    # Initial choices:
    init_controller = 'PID'
    init_mode = 'disturbance'

    # Run initial simulation
    t, state_history, control_force_history = run_simulation(controller_type=init_controller, mode=init_mode)

    # Create a figure for interactive simulation.
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.35)
    l_anim, = ax.plot(state_history[0, :], state_history[2, :]*180/np.pi, 'b-', lw=2)
    ax.set_title(f'Cart Position vs. Pendulum Angle ({init_controller} Controller, {init_mode} Mode)')
    ax.set_xlabel('Cart Position (m)')
    ax.set_ylabel('Pendulum Angle (deg)')
    ax.grid(True)

    # Create sliders for PID gains if PID controller is chosen.
    ax_kp = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_kd = plt.axes([0.25, 0.20, 0.65, 0.03])
    s_kp = Slider(ax_kp, 'Kp', 0.0, 200.0, valinit=100.0)
    s_kd = Slider(ax_kd, 'Kd', 0.0, 50.0, valinit=20.0)

    # Radio buttons to choose controller type
    ax_radio = plt.axes([0.025, 0.5, 0.15, 0.15])
    radio = RadioButtons(ax_radio, ('PID', 'Pole', 'Nonlinear'), active=0)

    # Radio buttons to choose simulation mode
    ax_mode = plt.axes([0.025, 0.7, 0.15, 0.15])
    radio_mode = RadioButtons(ax_mode, ('disturbance', 'moving'), active=0)

    def update(val):
        current_controller = radio.value_selected
        current_mode = radio_mode.value_selected
        if current_controller == 'PID':
            pid = PIDController(kp=s_kp.val, ki=0.0, kd=s_kd.val)
            controller = pid
        elif current_controller == 'Pole':
            controller = PolePlacementController()
        else:
            controller = NonlinearController()

        sensor = Sensor(noise_params)
        def dynamics_wrapper(t, state):
            derivs = dynamics(t, state, controller, sensor, params, disturbance, ref=[0,0], mode=current_mode)
            return derivs
        t_span = (sim_settings['t_start'], sim_settings['t_end'])
        t_eval = np.arange(sim_settings['t_start'], sim_settings['t_end'], sim_settings['dt'])
        sol = solve_ivp(dynamics_wrapper, t_span, [0.0, 0.0, np.deg2rad(2.0), 0.0], t_eval=t_eval, method='RK45')
        l_anim.set_data(sol.y[0, :], sol.y[2, :]*180/np.pi)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    s_kp.on_changed(update)
    s_kd.on_changed(update)
    radio.on_clicked(update)
    radio_mode.on_clicked(update)

    plt.show()

##########################################
# 8. MAIN FUNCTION
##########################################

def main():
    """
    Run the simulation and animation.
    You can change the controller type and mode below:
       controller_type: 'PID', 'Pole', 'Nonlinear'
       mode: 'disturbance' or 'moving'
    """
    # ----- Choose simulation configuration -----
    controller_type = 'PID'     # Change to 'Pole' or 'Nonlinear' to test other controllers.
    mode = 'disturbance'        # 'disturbance' applies external disturbances; 'moving' makes the cart follow a trajectory.
    # --------------------------------------------

    # Run simulation using ODE integration
    t, state_history, control_force_history = run_simulation(controller_type=controller_type, mode=mode)

    # Animate simulation with time-series plots.
    animate_simulation(t, state_history, control_force_history, controller_type, mode)

    # Uncomment the next line to try the interactive simulation with sliders.
    # interactive_simulation()

if __name__ == "__main__":
    main()
