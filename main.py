#!/usr/bin/env python3
"""
Inverted Pendulum with a single URDF 'cart_pendulum_wheels.urdf', 
which has:
  - chassis (link0)
  - 4 wheels (link1..4) each revolve about x
  - rod (link5) revolve about y from chassis top
  - bob (link6) fixed on rod top

We apply LQR control by directly applying a horizontal force on the chassis link 
to keep rod upright at x=0. 
The wheels physically spin if friction is set properly. 
Rod can swing ±90° about y. Bob never detaches.

Press 'q' to quit or close PyBullet window. 
Left-click to poke the system. 
"""

import pybullet as pb
import pybullet_data
import time, math, random
import numpy as np
import matplotlib.pyplot as plt

# If not defined:
MOUSE_BUTTON_LEFT = 0
KEY_WAS_TRIGGERED = 1

##############################################################################
# 1) Params
##############################################################################
class Params:
    def __init__(self):
        # LQR Gains: Force = -K [ (x- x_des), xdot, theta, thetadot ]
        self.x_des = 0.0
        self.K_lqr = np.array([-20., -5., 40., 7.5])

        # sensor noise
        self.use_noise = True
        self.noise_std_x     = 0.002
        self.noise_std_xdot  = 0.002
        self.noise_std_th    = 0.002
        self.noise_std_thdot = 0.002
        self.alpha_filter    = 0.9

        # control saturation
        self.use_saturation = True
        self.max_force = 50.0

        # disturbance
        self.disturb_time  = 2.0
        self.disturb_force = 5.0
        self.tip_angle_deg = 5.0

        # time
        self.dt = 1./240.
        self.gravity = 9.81

##############################################################################
# 2) Sensor
##############################################################################
class Sensor:
    def __init__(self, p:Params):
        self.p = p
        self.prev_meas = None

    def measure_and_filter(self, x, xdot, theta, thetadot):
        if self.p.use_noise:
            xn     = x     + random.gauss(0, self.p.noise_std_x)
            xdn    = xdot  + random.gauss(0, self.p.noise_std_xdot)
            thn    = theta + random.gauss(0, self.p.noise_std_th)
            thdn   = thetadot + random.gauss(0, self.p.noise_std_thdot)
        else:
            xn, xdn, thn, thdn = x, xdot, theta, thetadot

        meas = np.array([xn, xdn, thn, thdn])
        if self.prev_meas is None:
            self.prev_meas = meas
            return meas
        alpha= self.p.alpha_filter
        filtered= alpha*self.prev_meas + (1.-alpha)*meas
        self.prev_meas= filtered
        return filtered

##############################################################################
# 3) LQR Controller
##############################################################################
class Controller:
    def __init__(self, p:Params):
        self.p = p

    def compute_force(self, meas):
        x, xdot, theta, thetadot = meas
        e = np.array([ (x - self.p.x_des), xdot, theta, thetadot ])
        F = - self.p.K_lqr.dot(e)
        if self.p.use_saturation:
            F = np.clip(F, -self.p.max_force, self.p.max_force)
        return F

##############################################################################
# 4) Single URDF Simulation
##############################################################################
class InvertedPendulumSim:
    def __init__(self, p:Params):
        self.p= p
        self.sensor= Sensor(p)
        self.ctrl= Controller(p)

        self.cid= None
        self.body_id= None

        self.t_data= []
        self.x_data= []
        self.th_data= []
        self.f_data= []

        self.start_time= None
        self.last_time= None

    def setup_sim(self):
        self.cid = pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0,0, -self.p.gravity)
        pb.setRealTimeSimulation(0)

        # load plane
        plane_id= pb.loadURDF("plane.urdf", useFixedBase=True)
        # friction so wheels can roll
        pb.changeDynamics(plane_id, -1, lateralFriction=1.0, spinningFriction=1.0, rollingFriction=0.01)

        # load our single URDF
        self.body_id= pb.loadURDF("cart_pendulum_wheels.urdf",
                                  basePosition=[0,0,0],
                                  useFixedBase=False)

        # We'll disable motors on revolve wheel joints so they spin freely
        # Link indices: 0 => chassis, 1 => front_left_wheel, 2 => front_right_wheel, 3 => rear_left, 4 => rear_right,
        # 5 => rod, 6 => bob. Actually in pybullet the link indices are 0..6 for each joint? 
        # We'll do so by reading joint info:
        nJ= pb.getNumJoints(self.body_id)
        for j in range(nJ):
            jinfo= pb.getJointInfo(self.body_id, j)
            jName= jinfo[1].decode("utf-8")
            # for wheels revolve => set to VELOCITY_CONTROL, force=0 => free spin
            if "wheel" in jName:
                pb.setJointMotorControl2(self.body_id, j, 
                                         controlMode=pb.VELOCITY_CONTROL,
                                         targetVelocity=0,
                                         force=0)
            elif "pendulum" in jName:
                # rod revolve => also free
                pb.setJointMotorControl2(self.body_id, j,
                                         controlMode=pb.VELOCITY_CONTROL,
                                         targetVelocity=0,
                                         force=0)
            elif "bob" in jName:
                # fixed => no need
                pass

    def handle_mouse(self):
        events= pb.getMouseEvents()
        if not events:
            return
        cam= pb.getDebugVisualizerCamera()
        if len(cam)<11:
            return
        w= cam[0]; h= cam[1]
        view= cam[2]; proj= cam[3]

        for e in events:
            etype= e[0]
            state= e[1]
            mx= e[2]
            my= e[3]
            if (etype==MOUSE_BUTTON_LEFT) and (state & KEY_WAS_TRIGGERED):
                rf, rt= self.compute_ray(mx,my,w,h,view,proj)
                hits= pb.rayTest(rf,rt)
                if hits and hits[0][0]>=0:
                    bUid= hits[0][0]
                    fFrac= hits[0][2]
                    hitPos= [rf[i]+(rt[i]-rf[i])*fFrac for i in range(3)]
                    fMag= 5.0
                    fDir= [rt[i]-rf[i] for i in range(3)]
                    dist= math.sqrt(sum(ff*ff for ff in fDir))
                    if dist>1e-9:
                        invd=1.0/dist
                        fDir=[a*invd for a in fDir]
                        fVec=[fMag*a for a in fDir]
                        pb.applyExternalForce(bUid, -1, fVec, hitPos, pb.WORLD_FRAME)

    def compute_ray(self,mx,my,w,h,view,proj):
        ndcX= (mx-w/2.)/(w/2.)
        ndcY= (h/2.-my)/(h/2.)
        import numpy as np
        invV= np.linalg.inv(np.array(view).reshape((4,4)).T)
        invP= np.linalg.inv(np.array(proj).reshape((4,4)).T)
        nearPt= np.array([ndcX, ndcY, -1,1])
        farPt = np.array([ndcX, ndcY,  1,1])
        nearCam= invP.dot(nearPt); nearCam/= nearCam[3]
        farCam= invP.dot(farPt);   farCam/= farCam[3]
        nearWorld= invV.dot(nearCam); nearWorld/= nearWorld[3]
        farWorld = invV.dot(farCam);  farWorld/= farWorld[3]
        rayFrom= nearWorld[:3]
        rayTo=   farWorld[:3]
        return rayFrom, rayTo

    def get_states(self):
        """
        We'll read chassis base x, xdot, 
        and rod revolve joint for angle => see which link is the rod, 
        or we can read the joint with name 'joint_pendulum'.
        """
        # read chassis base
        basePos, baseOri= pb.getBasePositionAndOrientation(self.body_id)
        baseVel, baseAng= pb.getBaseVelocity(self.body_id)
        x= basePos[0]
        xdot= baseVel[0]

        # find pendulum joint
        nJ= pb.getNumJoints(self.body_id)
        pendulum_index= None
        for j in range(nJ):
            jinfo= pb.getJointInfo(self.body_id,j)
            jName= jinfo[1].decode("utf-8")
            if "pendulum" in jName:
                pendulum_index= j
                break
        if pendulum_index is None:
            # fallback, just pick link5?
            pendulum_index= 5

        jState= pb.getJointState(self.body_id, pendulum_index)
        theta= jState[0]
        thetadot= jState[1]

        return x, xdot, theta, thetadot

    def run(self):
        self.setup_sim()
        self.start_time= time.time()
        self.last_time= self.start_time

        while True:
            keys= pb.getKeyboardEvents()
            if ord('q') in keys and (keys[ord('q')] & KEY_WAS_TRIGGERED):
                break
            if not pb.getConnectionInfo().get('isConnected', True):
                break

            now= time.time()
            dt= now- self.last_time
            if dt< self.p.dt:
                time.sleep(self.p.dt- dt)
                now= time.time()
            self.last_time= now
            t_run= now- self.start_time

            # read states
            x, xdot, theta, thetadot = self.get_states()

            # measure
            meas= self.sensor.measure_and_filter(x, xdot, theta, thetadot)

            # LQR
            F_ctrl= self.ctrl.compute_force(meas)

            # Disturb at t=2
            F_extra= 0.0
            if abs(t_run- self.p.disturb_time)< 0.5*self.p.dt:
                F_extra+= self.p.disturb_force
                # forcibly tip the rod by tip_angle
                # We'll do setJointState or something
                # Actually for safety, we do getJointState -> new angle -> add
                nJ= pb.getNumJoints(self.body_id)
                for j in range(nJ):
                    jinfo= pb.getJointInfo(self.body_id, j)
                    if "pendulum" in jinfo[1].decode("utf-8"):
                        old_theta= pb.getJointState(self.body_id,j)[0]
                        new_theta= old_theta + math.radians(self.p.tip_angle_deg)
                        pb.resetJointState(self.body_id, j, new_theta, 0)
                        break

            F_total= F_ctrl+ F_extra

            # apply horizontal force to chassis link. linkIndex = -1 is the base
            pb.applyExternalForce(self.body_id, -1,
                                  forceObj=[F_total,0,0],
                                  posObj=[0,0,0],  # apply near center
                                  flags= pb.WORLD_FRAME)

            self.handle_mouse()
            pb.stepSimulation()

            self.t_data.append(t_run)
            self.x_data.append(x)
            self.th_data.append(theta)
            self.f_data.append(F_total)

        # done
        return np.array(self.t_data), np.array(self.x_data), np.array(self.th_data), np.array(self.f_data)

    def plot_results(self, t_log, x_log, th_log, f_log):
        fig, axs= plt.subplots(3,1, figsize=(8,6))
        axs[0].plot(t_log, x_log, 'b-', label='Cart X(m)')
        axs[0].grid(); axs[0].legend()

        axs[1].plot(t_log, np.degrees(th_log), 'r-', label='Pend Angle (deg) (hinge about y)')
        axs[1].grid(); axs[1].legend()

        axs[2].plot(t_log, f_log, 'm-', label='Control Force (N)')
        axs[2].grid(); axs[2].legend()

        plt.suptitle("Cart+Pendulum in single URDF w/ real wheels, LQR, press 'q' to quit")
        plt.tight_layout()
        plt.show()

def main():
    print("Single URDF with chassis, 4 wheels, rod, bob. LQR control. Press 'q' to quit.")
    from sys import platform
    p= Params()
    sim= InvertedPendulumSim(p)
    t_log, x_log, th_log, f_log= sim.run()
    print("Simulation ended. Plotting logs.")
    sim.plot_results(t_log, x_log, th_log, f_log)

if __name__=="__main__":
    main()
