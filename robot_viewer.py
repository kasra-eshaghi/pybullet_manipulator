import pybullet as p
import pybullet_data
import time
import math

def main():
    # Connect to the physics engine
    physicsClient = p.connect(p.GUI)
    
    # Add search path for the base URDFs (like plane)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load the ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Load the custom welding robot
    # useFixedBase=True anchors the base of the robot to the world
    print("Loading welding_robot.urdf...")
    try:
        robot_id = p.loadURDF("welding_robot.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        p.disconnect()
        return

    # Add basic GUI sliders to control the robot's joints
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot successfully loaded. Found {num_joints} joints/links.")
    
    joint_sliders = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        
        # We only care about revolute or prismatic joints (type 0 or 1), skip fixed joints (type 4)
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit >= upper_limit:
                lower_limit, upper_limit = -math.pi, math.pi
            
            slider_id = p.addUserDebugParameter(joint_name, lower_limit, upper_limit, 0)
            joint_sliders.append((i, slider_id))
    
    print("Use the PyBullet sliders to control the joints of the arm.")
    print("The welding torch should be attached to the end-effector (tool0).")
    
    # Set default camera position
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0.5])
    
    # Simulation loop
    try:
        while True:
            # Read sliders and update joint positions
            for joint_idx, slider_id in joint_sliders:
                target_pos = p.readUserDebugParameter(slider_id)
                # Position control for standard robot arm manipulation
                p.setJointMotorControl2(
                    bodyIndex=robot_id, 
                    jointIndex=joint_idx, 
                    controlMode=p.POSITION_CONTROL, 
                    targetPosition=target_pos,
                    force=1000.0,
                    maxVelocity=1.0
                )
            
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("Simulation terminated.")
    
    p.disconnect()

if __name__ == '__main__':
    main()
