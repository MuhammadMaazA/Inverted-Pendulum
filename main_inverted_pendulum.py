#!/usr/bin/env python
"""
Inverted Pendulum with separate URDFs (cart, rod, bob),
connected by point2point constraints. We do a PD or LQR controller,
sensor noise, and a mouse picking each frame so you can poke the system.

We run an indefinite loop:
 - The pendulum tries to maintain x_des (0 for scenario A, 0.3 for scenario B).
 - The window stays open until you close it or Ctrl+C.

Cart & rod are placed so the rod is upright and the bob is on top, 
thanks to offsets in load_* calls. 
"""

import pybullet as pb
import pybullet_data
import math
import time
import random
import numpy as np

###############################################################################
# 1) Parameters & Sensor
###############################################################################
class Params:
    def __init__(self):
        # "scenario": either 0 => x=0, or 0.3 => x=0.3
        self.x_des = 0.0    # default scenario = A
        self.controller_type = "PD"  # or "LQR"

        # Gains PD
        self.Kp_x  = 50.0
        self.Kd_x  = 10.0
        self.Kp_th = 100.0
        self.Kd_th = 10.0

        # Gains LQR: F = -K [ x - x_des, xdot, th, thdot]
        self.K_lqr = np.array([-40., -10., 80., 15.])

        # sensor noise
        self.use_noise = True
        self.noise_std_x      = 0.002
        self.noise_std_xdot   = 0.002
        self.noise_std_th     = 0.002
        self.noise_std_thdot  = 0.002
        self.alpha_filter     = 0.9

        # optional saturation
        self.use_saturation = False
        self.max_force = 50.0

        # For Disturbances if desired
        self.disturb_time = 2.0
        self.disturb_force= 5.0
        self.tip_angle_deg= 5.0

        # indefinite loop step
        self.dt = 1./240.
        self.gravity = 9.81

class Sensor:
    def __init__(self, p:Params):
        self.p = p
        self.prev_meas = None

    def measure_and_filter(self, x, xdot, th, thdot):
        if self.p.use_noise:
            xn    = x     + random.gauss(0, self.p.noise_std_x)
            xdn   = xdot  + random.gauss(0, self.p.noise_std_xdot)
            thn   = th    + random.gauss(0, self.p.noise_std_th)
            thdn  = thdot + random.gauss(0, self.p.noise_std_thdot)
        else:
            xn, xdn, thn, thdn = x, xdot, th, thdot

        meas = np.array([xn, xdn, thn, thdn])
        if self.prev_meas is None:
            self.prev_meas = meas
            return meas
        filtered = self.p.alpha_filter*self.prev_meas + (1.0 - self.p.alpha_filter)*meas
        self.prev_meas = filtered
        return filtered

###############################################################################
# 2) Controllers
###############################################################################
def controller_PD(p:Params, meas):
    x, xdot, th, thdot = meas
    ex = (p.x_des - x)
    edotx = -xdot
    eth = (0. - th)
    edotth= -thdot

    Fx  = p.Kp_x*ex + p.Kd_x*edotx
    Fth = p.Kp_th*eth+ p.Kd_th*edotth
    F = Fx + Fth

    if p.use_saturation:
        F = np.clip(F, -p.max_force, p.max_force)
    return F

def controller_LQR(p:Params, meas):
    x, xdot, th, thdot = meas
    e = np.array([ (x - p.x_des), xdot, th, thdot])
    F = - p.K_lqr.dot(e)
    if p.use_saturation:
        F = np.clip(F, -p.max_force, p.max_force)
    return F

###############################################################################
# 3) Mouse picking: left-click => small impulse from camera ray
###############################################################################
def check_mouse_and_apply_force():
    events = pb.getMouseEvents()
    if len(events)==0:
        return
    w,h,view,proj,camUp,camFwd,hor,ver,yaw,pitch,camTarget=pb.getDebugVisualizerCamera()
    for e in events:
        etype=e[0]
        state=e[1]
        mx=e[2]
        my=e[3]
        if (etype==pb.MOUSE_BUTTON_LEFT) and (state & pb.KEY_WAS_TRIGGERED):
            rayFrom, rayTo= computeRay(mx,my,w,h,view,proj)
            hits= pb.rayTest(rayFrom, rayTo)
            if len(hits)>0:
                bodyUid= hits[0][0]
                hitFrac= hits[0][2]
                if bodyUid>=0:
                    hitPos= [rayFrom[i]+(rayTo[i]-rayFrom[i])*hitFrac for i in range(3)]
                    forceMag= 5.0
                    forceDir= [rayTo[i]-rayFrom[i] for i in range(3)]
                    dist= math.sqrt(sum(ff*ff for ff in forceDir))
                    if dist>1e-9:
                        invd=1./dist
                        forceDir=[f*invd for f in forceDir]
                        forceVec=[forceMag*f for f in forceDir]
                        pb.applyExternalForce(bodyUid, -1,
                                             forceObj=forceVec,
                                             posObj=hitPos,
                                             flags=pb.WORLD_FRAME)

def computeRay(mx,my,w,h,view,proj):
    ndcX= (mx - w/2.)/(w/2.)
    ndcY= (h/2. - my)/(h/2.)
    import numpy as np
    invV= np.linalg.inv(np.array(view).reshape((4,4)).T)
    invP= np.linalg.inv(np.array(proj).reshape((4,4)).T)
    nearPt= np.array([ndcX, ndcY, -1,1])
    farPt = np.array([ndcX, ndcY, +1,1])

    nearCam= invP.dot(nearPt); nearCam/= nearCam[3]
    farCam= invP.dot(farPt);   farCam /= farCam[3]
    nearWorld= invV.dot(nearCam); nearWorld/= nearWorld[3]
    farWorld = invV.dot(farCam);  farWorld /= farWorld[3]
    rayFrom= nearWorld[:3]
    rayTo= farWorld[:3]
    return rayFrom, rayTo

###############################################################################
# 4) Load URDFs, using point2point constraints
###############################################################################
def load_cart_rod_bob():
    """
    We'll carefully place the rod so that it lines up upright 
    and the bob on top. We'll do point2point constraints.
    """
    cart_id= pb.loadURDF("cart.urdf", basePosition=[0,0,0], useFixedBase=False)
    # rod: center-based if length=0.45 => top=+0.225, bottom=-0.225
    # we want bottom near cart top => cart is 0.1 tall => so rod center=0.325
    rod_id= pb.loadURDF("rod.urdf", basePosition=[0,0,0.325],useFixedBase=False)
    # bob => at rod top => rod center=0.325 => top=0.325+0.225=0.55
    bob_id= pb.loadURDF("bob.urdf", basePosition=[0,0,0.55],useFixedBase=False)

    # create p2p from cart top => rod bottom
    # cart top is [0,0,0.1], 
    # rod bottom is local -0.225 from center => global=0.325 - 0.225=0.1 => perfect
    c1= pb.createConstraint(
        parentBodyUniqueId=cart_id, parentLinkIndex=-1,
        childBodyUniqueId=rod_id, childLinkIndex=-1,
        jointType=pb.JOINT_POINT2POINT,
        jointAxis=[0,0,0],
        parentFramePosition=[0,0,0.1],
        childFramePosition=[0,0,-0.225]
    )
    # rod->bob
    # rod top => local is +0.225 => global=0.55
    c2= pb.createConstraint(
        parentBodyUniqueId=rod_id, parentLinkIndex=-1,
        childBodyUniqueId=bob_id, childLinkIndex=-1,
        jointType=pb.JOINT_POINT2POINT,
        jointAxis=[0,0,0],
        parentFramePosition=[0,0,0.225],
        childFramePosition=[0,0,0]
    )
    return cart_id, rod_id, bob_id

###############################################################################
# 5) Indefinite run
###############################################################################
def run_indefinite(params:Params):
    cid= pb.connect(pb.GUI)
    import pybullet_data
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0,0,-params.gravity)
    pb.setRealTimeSimulation(0)

    plane_id= pb.loadURDF("plane.urdf")
    cart_id,rod_id,bob_id= load_cart_rod_bob()

    # We'll do a sensor + logs
    sensor= Sensor(params)
    # store logs for plotting if you want
    # But we do indefinite => let's store up to e.g. 1000 steps
    t_data=[]
    x_data=[]
    th_data=[]
    f_data=[]

    i=0
    startTime= time.time()
    lastTime= startTime
    while True:
        now= time.time()
        dt= now-lastTime
        # we want a stable dt => let's just do param dt
        if dt< params.dt:
            time.sleep(params.dt- dt)
            now= time.time()
        dt2= now- lastTime
        lastTime= now
        t_run= now- startTime

        # read cart
        cart_pos, cart_ori= pb.getBasePositionAndOrientation(cart_id)
        cart_vel, cart_ang= pb.getBaseVelocity(cart_id)
        x_= cart_pos[0]
        xdot_= cart_vel[0]

        # rod
        rod_pos, rod_ori= pb.getBasePositionAndOrientation(rod_id)
        rod_vel, rod_ang= pb.getBaseVelocity(rod_id)
        eul_rod= pb.getEulerFromQuaternion(rod_ori)
        th_= eul_rod[0]
        thdot_= rod_ang[1]

        # measure
        meas= sensor.measure_and_filter(x_, xdot_, th_, thdot_)

        # control
        if params.controller_type.upper()=="PD":
            F_ctrl= controller_PD(params, meas)
        else:
            F_ctrl= controller_LQR(params, meas)

        # Disturb at t=2
        F_extra=0.0
        if abs(t_run- params.disturb_time)< 0.5*params.dt:
            F_extra+= params.disturb_force
            old_eul= list(eul_rod)
            old_eul[0]+= math.radians(params.tip_angle_deg)
            new_q= pb.getQuaternionFromEuler(old_eul)
            pb.resetBasePositionAndOrientation(rod_id, rod_pos, new_q)

        F_total= F_ctrl+F_extra
        pb.applyExternalForce(cart_id, -1,
                              forceObj=[F_total,0,0],
                              posObj=cart_pos,
                              flags= pb.WORLD_FRAME)

        check_mouse_and_apply_force()
        pb.stepSimulation()

        # store logs
        t_data.append(t_run)
        x_data.append(x_)
        th_data.append(th_)
        f_data.append(F_total)

        i+=1
        # We'll do indefinite unless user closes window
        # If we want to see if user closed, we can check connectionInfo
        if not pb.getConnectionInfo()['isConnected']:
            print("PyBullet window closed. Exiting loop.")
            break

    # at the end, we can do a quick plot
    # or just end
    return np.array(t_data), np.array(x_data), np.array(th_data), np.array(f_data)

###############################################################################
def main():
    p= Params()
    # Let the user pick scenario
    # scenario A => x_des=0
    # scenario B => x_des=0.3
    scenario= input("Pick scenario [A or B]? Default=A => x=0, B => x=0.3: ")
    if scenario.upper()=="B":
        p.x_des= 0.3
        scenario_name="Scenario B: x=0.3"
    else:
        p.x_des= 0.0
        scenario_name="Scenario A: x=0.0"

    # also pick "PD" or "LQR"
    ctype= input("Pick controller [PD or LQR]? Default=PD: ")
    if ctype.upper()=="LQR":
        p.controller_type="LQR"
    else:
        p.controller_type="PD"

    print(f"Running indefinite with {scenario_name}, controller={p.controller_type}.\n")
    t_log, x_log, th_log, f_log= run_indefinite(p)
    print("Simulation ended (PyBullet window closed or user ctrl+c). Plotting...")

    import matplotlib.pyplot as plt
    fig, axs= plt.subplots(3,1, figsize=(7,7))
    axs[0].plot(t_log, x_log, 'b-', label='Cart X(m)')
    axs[0].grid(); axs[0].legend()

    axs[1].plot(t_log, np.degrees(th_log), 'r-', label='Rod Angle(deg approx)')
    axs[1].grid(); axs[1].legend()

    axs[2].plot(t_log, f_log, 'm-', label='Force + Disturb')
    axs[2].grid(); axs[2].legend()

    plt.suptitle(f"{scenario_name}, {p.controller_type} control, indefinite run logs")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
