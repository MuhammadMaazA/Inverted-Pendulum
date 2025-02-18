import pybullet as p
import pybullet_data
import time
import os

def main():
    # Connect to PyBullet in GUI mode
    p.connect(p.GUI)

    # Add PyBullet's default search path (plane.urdf, etc.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    p.setGravity(0, 0, -9.81)

    # Load a ground plane
    plane_id = p.loadURDF("plane.urdf")

    # Load the cart-pole URDF from current folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(current_dir)
    cart_id = p.loadURDF("flagpole.urdf", [0, 0, 0.1])

    # Disable all motor controls (open-loop)
    num_joints = p.getNumJoints(cart_id)
    for j in range(num_joints):
        # VELOCITY_CONTROL with force=0 means "no torque"
        p.setJointMotorControl2(
            bodyUniqueId=cart_id,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0
        )

    # Optionally add some friction so it doesn't slide forever
    for link_idx in range(-1, num_joints):
        p.changeDynamics(cart_id, link_idx, lateralFriction=1.0)

    # Simulation loop
    dt = 1.0 / 240.0
    print("Open-loop cart-pole. Poke the bob or rod in the GUI to see it topple!")
    while True:
        p.stepSimulation()
        time.sleep(dt)

if __name__ == "__main__":
    main()
