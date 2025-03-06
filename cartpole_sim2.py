import pybullet as p
import pybullet_data
import numpy as np
import time
from controllers import PIDController


def main():
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    cartpole = p.loadURDF("flagpole.urdf", [0, 0, 0.1])

    num_joints = p.getNumJoints(cartpole)
    pole_joint = None
    wheel_joints = []
    for j in range(num_joints):
        info = p.getJointInfo(cartpole, j)
        joint_name = info[1].decode('utf-8')
        if joint_name == 'pole_base_joint':
            pole_joint = j
        elif 'wheel' in joint_name:
            wheel_joints.append(j)

    controller = PIDController(Kp=100, Ki=1, Kd=20)
    dt = 1/240

    for step in range(2400):
        pole_angle = p.getJointState(cartpole, pole_joint)[0]
        pole_angle_noisy = pole_angle + np.random.normal(0, 0.01)

        control_force = controller.compute(pole_angle_noisy, dt)
        control_force = np.clip(control_force, -10, 10)

        for wheel_joint in wheel_joints:
            p.setJointMotorControl2(cartpole, wheel_joint, p.TORQUE_CONTROL, force=control_force)

        if step == 1200:
            p.applyExternalForce(cartpole, -1, [50, 0, 0], [0, 0, 0], p.WORLD_FRAME)

        p.stepSimulation()
        time.sleep(dt)

    p.disconnect()


if __name__ == '__main__':
    main()
