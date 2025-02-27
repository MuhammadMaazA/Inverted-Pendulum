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

    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")

    # Load the frictionless-joint cart+rod URDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(current_dir)
    cart_id = p.loadURDF("flagpole_decoupled.urdf", [0, 0, 0.1], useFixedBase=False)

    # Optionally add friction so the cart can move if poked
    # but doesn't slide forever.
    num_joints = p.getNumJoints(cart_id)
    for link_idx in range(-1, num_joints):
        p.changeDynamics(cart_id, link_idx, lateralFriction=1.0)

    dt = 1.0 / 240.0
    print("Poke the rod -> it rotates in place, not moving the cart.")
    print("Poke the cart -> cart moves, rod translates with cart's base.")
    while True:
        p.stepSimulation()
        time.sleep(dt)

if __name__ == "__main__":
    main()
