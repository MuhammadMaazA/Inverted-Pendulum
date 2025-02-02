#!/usr/bin/env python3
"""
Inverted Pendulum Simulation using PyBullet with an LQR Controller.
This simulation uses three URDF files (cart, rod, bob) located in the same directory.
The system starts in a stable configuration (cart at x = 0; pendulum upright)
and remains running until you press the 'q' key (while the window is active) or manually close the GUI.
You can interact with the simulation by left-clicking ("poking" the system); the controller will
try to restore stability.
"""

import pybullet as pb
import pybullet_data
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Define our own keyboard/mouse event constants (if not provided)
# ------------------------------------------------------------------
MOUSE_BUTTON_LEFT = 0      # left mouse button
KEY_WAS_TRIGGERED = 1      # flag for a key-press event

# =============================================================================
# PARAMETERS CLASS
# =============================================================================
class Params:
    def __init__(self):
        # Target: keep cart at x = 0
        self.x_des = 0.0
        
        # Use LQR controller (only one controller is used)
        # Control law: F = -K * [x - x_des, xdot, theta, theta_dot]
        # Gains tuned to be moderately aggressive.
        self.K_lqr = np.array([-20.0, -5.0, 40.0, 7.5])
        
        # Sensor noise parameters (optional)
        self.use_noise = True
        self.noise_std_x = 0.002
        self.noise_std_xdot = 0.002
        self.noise_std_th = 0.002
        self.noise_std_thdot = 0.002
        self.alpha_filter = 0.9

        # Control saturation (clip force output)
        self.use_saturation = True
        self.max_force = 50.0

        # Disturbance parameters (simulate an external push)
        self.disturb_time = 2.0       # seconds when a disturbance is applied
        self.disturb_force = 5.0      # extra force at disturbance
        self.tip_angle_deg = 5.0      # extra tip (in degrees) applied

        # Simulation parameters
        self.dt = 1.0 / 240.0
        self.gravity = 9.81

# =============================================================================
# SENSOR CLASS
# =============================================================================
class Sensor:
    def __init__(self, params: Params):
        self.params = params
        self.prev_meas = None

    def measure_and_filter(self, x, xdot, th, thdot):
        p = self.params
        if p.use_noise:
            xn    = x + random.gauss(0, p.noise_std_x)
            xdn   = xdot + random.gauss(0, p.noise_std_xdot)
            thn   = th + random.gauss(0, p.noise_std_th)
            thdn  = thdot + random.gauss(0, p.noise_std_thdot)
        else:
            xn, xdn, thn, thdn = x, xdot, th, thdot

        meas = np.array([xn, xdn, thn, thdn])
        if self.prev_meas is None:
            self.prev_meas = meas
            return meas
        filtered = p.alpha_filter * self.prev_meas + (1.0 - p.alpha_filter) * meas
        self.prev_meas = filtered
        return filtered

# =============================================================================
# CONTROLLER CLASS (LQR)
# =============================================================================
class Controller:
    def __init__(self, params: Params):
        self.params = params

    def compute_force(self, meas):
        p = self.params
        x, xdot, th, thdot = meas
        error = np.array([x - p.x_des, xdot, th, thdot])
        F = -p.K_lqr.dot(error)
        if p.use_saturation:
            F = np.clip(F, -p.max_force, p.max_force)
        return F

# =============================================================================
# INVERTED PENDULUM SIMULATION CLASS
# =============================================================================
class InvertedPendulumSim:
    def __init__(self, params: Params):
        self.params = params
        self.sensor = Sensor(params)
        self.controller = Controller(params)

        # Data logging
        self.t_data = []
        self.x_data = []
        self.th_data = []
        self.f_data = []

        self.start_time = None
        self.last_time = None

        self.cart_id = None
        self.rod_id = None
        self.bob_id = None
        self.cid = None

    def setup_simulation(self):
        self.cid = pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -self.params.gravity)
        pb.setRealTimeSimulation(0)
        pb.loadURDF("plane.urdf")
        self.load_models()

    def load_models(self):
        # Load URDF files (assumed to be in the same directory)
        self.cart_id = pb.loadURDF("cart.urdf", basePosition=[0, 0, 0], useFixedBase=False)
        self.rod_id  = pb.loadURDF("rod.urdf", basePosition=[0, 0, 0.325], useFixedBase=False)
        self.bob_id  = pb.loadURDF("bob.urdf", basePosition=[0, 0, 0.55], useFixedBase=False)

        # (Optional) Apply damping to reduce aggressive dynamics.
        pb.changeDynamics(self.cart_id, -1, linearDamping=0.1, angularDamping=0.1)
        pb.changeDynamics(self.rod_id,  -1, linearDamping=0.1, angularDamping=0.1)
        pb.changeDynamics(self.bob_id,  -1, linearDamping=0.1, angularDamping=0.1)

        # Create a point-to-point constraint between the cart and the rod.
        # Attach the cart's top ([0, 0, 0.1]) to the rod's bottom (local [0, 0, -0.225]).
        c1 = pb.createConstraint(
            parentBodyUniqueId=self.cart_id, parentLinkIndex=-1,
            childBodyUniqueId=self.rod_id, childLinkIndex=-1,
            jointType=pb.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.1],
            childFramePosition=[0, 0, -0.225]
        )
        pb.changeConstraint(c1, maxForce=1000)

        # Create a fixed constraint between the rod and the bob so that the bob stays attached.
        # Attach the rod's top (local [0, 0, 0.225]) to the bob's center ([0, 0, 0]).
        c2 = pb.createConstraint(
            parentBodyUniqueId=self.rod_id, parentLinkIndex=-1,
            childBodyUniqueId=self.bob_id, childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.225],
            childFramePosition=[0, 0, 0]
        )
        # For a fixed joint, no need to change constraint forces

    def compute_mouse_force(self):
        """Checks for left mouse clicks and applies a small impulse to the clicked object."""
        events = pb.getMouseEvents()
        if not events:
            return

        # Get camera parameters (we assume at least 11 values are returned)
        camera_info = pb.getDebugVisualizerCamera()
        if len(camera_info) < 11:
            return
        w, h, view, proj, camUp, camFwd, hor, ver, yaw, pitch, camTarget = camera_info[:11]

        for e in events:
            etype, state, mx, my, *_ = e
            if (etype == MOUSE_BUTTON_LEFT) and (state & KEY_WAS_TRIGGERED):
                rayFrom, rayTo = self.compute_ray(mx, my, w, h, view, proj)
                hits = pb.rayTest(rayFrom, rayTo)
                if hits and hits[0][0] >= 0:
                    bodyUid = hits[0][0]
                    hitFrac = hits[0][2]
                    hitPos = [rayFrom[i] + (rayTo[i] - rayFrom[i]) * hitFrac for i in range(3)]
                    forceMag = 5.0
                    forceDir = [rayTo[i] - rayFrom[i] for i in range(3)]
                    dist = math.sqrt(sum(f * f for f in forceDir))
                    if dist > 1e-9:
                        invd = 1.0 / dist
                        forceDir = [f * invd for f in forceDir]
                        forceVec = [forceMag * f for f in forceDir]
                        pb.applyExternalForce(bodyUid, -1, forceObj=forceVec,
                                              posObj=hitPos, flags=pb.WORLD_FRAME)

    def compute_ray(self, mx, my, w, h, view, proj):
        ndcX = (mx - w / 2.0) / (w / 2.0)
        ndcY = (h / 2.0 - my) / (h / 2.0)
        invV = np.linalg.inv(np.array(view).reshape((4, 4)).T)
        invP = np.linalg.inv(np.array(proj).reshape((4, 4)).T)
        nearPt = np.array([ndcX, ndcY, -1, 1])
        farPt  = np.array([ndcX, ndcY, 1, 1])
        nearCam = invP.dot(nearPt)
        nearCam /= nearCam[3]
        farCam  = invP.dot(farPt)
        farCam  /= farCam[3]
        nearWorld = invV.dot(nearCam)
        nearWorld /= nearWorld[3]
        farWorld  = invV.dot(farCam)
        farWorld  /= farWorld[3]
        rayFrom = nearWorld[:3]
        rayTo   = farWorld[:3]
        return rayFrom, rayTo

    def run(self):
        self.setup_simulation()
        self.start_time = time.time()
        self.last_time = self.start_time

        # Run indefinitely until you press 'q' (with the window active) or close the GUI.
        while True:
            keys = pb.getKeyboardEvents()
            if ord('q') in keys and (keys[ord('q')] & KEY_WAS_TRIGGERED):
                break
            if not pb.getConnectionInfo().get('isConnected', True):
                break

            now = time.time()
            dt = now - self.last_time
            if dt < self.params.dt:
                time.sleep(self.params.dt - dt)
                now = time.time()
            self.last_time = now
            t_run = now - self.start_time

            # --- Get system states ---
            # Cart state
            cart_pos, _ = pb.getBasePositionAndOrientation(self.cart_id)
            cart_vel, _ = pb.getBaseVelocity(self.cart_id)
            x = cart_pos[0]
            xdot = cart_vel[0]

            # Rod state
            rod_pos, rod_ori = pb.getBasePositionAndOrientation(self.rod_id)
            rod_vel, rod_ang = pb.getBaseVelocity(self.rod_id)
            eul_rod = pb.getEulerFromQuaternion(rod_ori)
            th = eul_rod[0]
            thdot = rod_ang[1]

            # --- Sensor measurement and control ---
            meas = self.sensor.measure_and_filter(x, xdot, th, thdot)
            F_ctrl = self.controller.compute_force(meas)

            # --- Apply disturbance (simulate an external push) ---
            F_extra = 0.0
            if abs(t_run - self.params.disturb_time) < 0.5 * self.params.dt:
                F_extra += self.params.disturb_force
                old_eul = list(eul_rod)
                old_eul[0] += math.radians(self.params.tip_angle_deg)
                new_q = pb.getQuaternionFromEuler(old_eul)
                pb.resetBasePositionAndOrientation(self.rod_id, rod_pos, new_q)

            F_total = F_ctrl + F_extra
            pb.applyExternalForce(self.cart_id, -1, forceObj=[F_total, 0, 0],
                                  posObj=cart_pos, flags=pb.WORLD_FRAME)

            # --- Process mouse events (to "poke" the system) ---
            self.compute_mouse_force()

            pb.stepSimulation()

            # --- Log data ---
            self.t_data.append(t_run)
            self.x_data.append(x)
            self.th_data.append(th)
            self.f_data.append(F_total)

        return (np.array(self.t_data), np.array(self.x_data),
                np.array(self.th_data), np.array(self.f_data))

    def plot_results(self, t_log, x_log, th_log, f_log):
        fig, axs = plt.subplots(3, 1, figsize=(7, 7))
        axs[0].plot(t_log, x_log, 'b-', label='Cart Position (m)')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(t_log, np.degrees(th_log), 'r-', label='Rod Angle (deg)')
        axs[1].grid()
        axs[1].legend()

        axs[2].plot(t_log, f_log, 'm-', label='Control Force (N)')
        axs[2].grid()
        axs[2].legend()

        plt.suptitle("Inverted Pendulum Simulation (LQR Controller)")
        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    # Fixed settings: target x_des = 0, use LQR controller.
    params = Params()
    params.x_des = 0.0
    params.controller_type = "LQR"
    print("Running simulation with x_des = 0 using LQR controller.")
    sim = InvertedPendulumSim(params)
    t_log, x_log, th_log, f_log = sim.run()
    print("Simulation ended (window closed or 'q' pressed). Plotting logs...")
    sim.plot_results(t_log, x_log, th_log, f_log)

if __name__ == "__main__":
    main()
