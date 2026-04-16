import pybullet as p
import pybullet_data
import time
import math

def draw_coordinate_frame(pos, orn, line_ids=None):
    """
    Draws a coordinate frame (RGB arrows) at the specified position and orientation.
    Returns the IDs of the lines so they can be updated instead of redrawn.
    """
    if line_ids is None:
        line_ids = [-1, -1, -1]
        
    # Convert quaternion to rotation matrix
    rot_matrix = p.getMatrixFromQuaternion(orn)
    rot_matrix = [rot_matrix[0:3], rot_matrix[3:6], rot_matrix[6:9]]
    
    length = 0.1
    # X, Y, Z axes local to the frame
    x_axis = [pos[0] + rot_matrix[0][0]*length, pos[1] + rot_matrix[1][0]*length, pos[2] + rot_matrix[2][0]*length]
    y_axis = [pos[0] + rot_matrix[0][1]*length, pos[1] + rot_matrix[1][1]*length, pos[2] + rot_matrix[2][1]*length]
    z_axis = [pos[0] + rot_matrix[0][2]*length, pos[1] + rot_matrix[1][2]*length, pos[2] + rot_matrix[2][2]*length]
    
    line_ids[0] = p.addUserDebugLine(pos, x_axis, [1, 0, 0], 2, replaceItemUniqueId=line_ids[0]) # X = Red
    line_ids[1] = p.addUserDebugLine(pos, y_axis, [0, 1, 0], 2, replaceItemUniqueId=line_ids[1]) # Y = Green
    line_ids[2] = p.addUserDebugLine(pos, z_axis, [0, 0, 1], 2, replaceItemUniqueId=line_ids[2]) # Z = Blue
    
    return line_ids


def main():
    # Connect
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")
    
    # Load Robot
    try:
        robot_id = p.loadURDF("welding_robot.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        p.disconnect()
        return

    # Find the tcp_link index and joint indices
    num_joints = p.getNumJoints(robot_id)
    tcp_link_index = -1
    dof_joint_indices = []
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")
        link_name = joint_info[12].decode("utf-8")
        joint_type = joint_info[2]
        
        if link_name == "tcp_link":
            tcp_link_index = i
            
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            dof_joint_indices.append(i)

    if tcp_link_index == -1:
        print("ERROR: Could not find 'tcp_link' in the URDF!")
        p.disconnect()
        return
        
    print(f"Found tcp_link at index {tcp_link_index}")

    # Initialize the robot to a reasonable stance so it doesn't solve IK from singular 0,0,0
    initial_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0] # Standard UR ready pose
    for i, joint_idx in enumerate(dof_joint_indices):
        if i < len(initial_joint_positions):
            p.resetJointState(robot_id, joint_idx, initial_joint_positions[i])

    # Get the current TCP pose to initialize the sliders
    tcp_state = p.getLinkState(robot_id, tcp_link_index)
    init_pos = tcp_state[4] # linkWorldPosition
    init_orn = tcp_state[5] # linkWorldOrientation
    init_euler = p.getEulerFromQuaternion(init_orn)

    # Create XYZ setup sliders around current TCP pose
    pos_x_slider = p.addUserDebugParameter("Target X", init_pos[0] - 1.0, init_pos[0] + 1.0, init_pos[0])
    pos_y_slider = p.addUserDebugParameter("Target Y", init_pos[1] - 1.0, init_pos[1] + 1.0, init_pos[1])
    pos_z_slider = p.addUserDebugParameter("Target Z", init_pos[2] - 1.0, init_pos[2] + 1.0, init_pos[2])

    roll_slider = p.addUserDebugParameter("Target Roll", -math.pi, math.pi, init_euler[0])
    pitch_slider = p.addUserDebugParameter("Target Pitch", -math.pi, math.pi, init_euler[1])
    yaw_slider = p.addUserDebugParameter("Target Yaw", -math.pi, math.pi, init_euler[2])

    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=init_pos)

    target_frame_ids = None

    try:
        while True:
            # Read targets from GUI
            tx = p.readUserDebugParameter(pos_x_slider)
            ty = p.readUserDebugParameter(pos_y_slider)
            tz = p.readUserDebugParameter(pos_z_slider)
            
            tr = p.readUserDebugParameter(roll_slider)
            tp = p.readUserDebugParameter(pitch_slider)
            tyaw = p.readUserDebugParameter(yaw_slider)
            
            target_pos = [tx, ty, tz]
            target_orn = p.getQuaternionFromEuler([tr, tp, tyaw])
            
            # Draw coordinate frame representing the target pose
            target_frame_ids = draw_coordinate_frame(target_pos, target_orn, target_frame_ids)

            # Solve Inverse Kinematics
            ik_solution = p.calculateInverseKinematics(
                robot_id, 
                tcp_link_index, 
                target_pos, 
                target_orn,
                maxNumIterations=100,
                residualThreshold=1e-4
            )
            
            # The IK solver returns a list corresponding to all Degrees of Freedom. 
            # We must apply it cleanly to the revolute joints we mapped.
            for i, joint_idx in enumerate(dof_joint_indices):
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=ik_solution[i],
                    force=500.0,
                    maxVelocity=1.0
                )
            
            p.stepSimulation()
            time.sleep(1/240.)
            
    except KeyboardInterrupt:
        pass
        
    p.disconnect()

if __name__ == '__main__':
    main()
