import pybullet as p
import pybullet_data
import pybullet_industrial as pi
import numpy as np
import time
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Welding Robot Trajectory Simulator")
    parser.add_argument('--tolerance', type=float, default=10.0, help="Maximum allowed IK tracking deviation in mm (default: 10.0)")
    args = parser.parse_args()
    tolerance_m = args.tolerance / 1000.0

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")
    
    # Enable Self Collision to allow the checker to register internal arm hits
    print("Loading URDF... Please wait")
    try:
        robot_id = p.loadURDF("welding_robot.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        p.disconnect()
        return

    num_joints = p.getNumJoints(robot_id)
    tcp_link_index = -1
    dof_joint_indices = []
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        link_name = joint_info[12].decode("utf-8")
        if link_name == "tcp_link":
            tcp_link_index = i
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            dof_joint_indices.append(i)

    # We will test three common configurations (seeds) for UR10 IK
    ik_seeds = [
        [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],  # Standard Elbow Up
        [0.0, -1.57, -1.57, -1.57, 1.57, 0.0],  # Elbow Down
        [3.14, -1.57, 1.57, -1.57, -1.57, 0.0], # Shoulder Flipped
    ]

    # Initialize robot strictly to the first seed
    for i, j in enumerate(dof_joint_indices):
        p.resetJointState(robot_id, j, ik_seeds[0][i])

    # Initialize pybullet_industrial Collision Checker
    collision_checker = pi.CollisionChecker([robot_id, plane_id])
    
    # Very important: Pybullet native meshes overlap slightly at the joints in the UR10 (e.g., wrist_3 hits the torch, base hits the floor).
    # Calling set_safe_state() marks the current non-exploding resting pose as structurally safe and whitelists these adjacent overlapping links.
    collision_checker.set_safe_state()
    
    # Control Panel
    pos_sx = p.addUserDebugParameter("Start X", -1.0, 1.0, 0.4)
    pos_sy = p.addUserDebugParameter("Start Y", -1.0, 1.0, 0.4)
    pos_sz = p.addUserDebugParameter("Start Z",  0.0, 1.5, 0.2)
    
    pos_ex = p.addUserDebugParameter("End X", -1.0, 1.0, 0.4)
    pos_ey = p.addUserDebugParameter("End Y", -1.0, 1.0, -0.4)
    pos_ez = p.addUserDebugParameter("End Z",  0.0, 1.5, 0.2)

    roll_slider = p.addUserDebugParameter("TCP Roll", -3.14, 3.14, 3.14)
    pitch_slider = p.addUserDebugParameter("TCP Pitch", -3.14, 3.14, 0.0)
    yaw_slider = p.addUserDebugParameter("TCP Yaw", -3.14, 3.14, 0.0)

    speed_slider = p.addUserDebugParameter("Speed (m/s)", 0.01, 1.0, 0.1)

    btn_exec = p.addUserDebugParameter("EXECUTE TRAJECTORY", 1, 0, 0)
    btn_cnc = p.addUserDebugParameter("CANCEL TRAJECTORY", 1, 0, 0)

    exec_clicks = 0
    cnc_clicks = 0

    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.4, 0, 0.2])

    last_params = None
    running_trajectory = False
    trajectory_points = []
    traj_idx = 0

    print("\n--- READY ---")
    print("Move sliders to define your Trajectory. Press EXECUTE when ready.\n")

    try:
        while True:
            current_exec = p.readUserDebugParameter(btn_exec)
            current_cnc = p.readUserDebugParameter(btn_cnc)

            # Check for CANCEL
            if current_cnc > cnc_clicks:
                cnc_clicks = current_cnc
                if running_trajectory:
                    print("--> Execution CANCELLED.")
                    running_trajectory = False

            # Check sliders
            sx, sy, sz = p.readUserDebugParameter(pos_sx), p.readUserDebugParameter(pos_sy), p.readUserDebugParameter(pos_sz)
            ex, ey, ez = p.readUserDebugParameter(pos_ex), p.readUserDebugParameter(pos_ey), p.readUserDebugParameter(pos_ez)
            
            roll, pitch, yaw = p.readUserDebugParameter(roll_slider), p.readUserDebugParameter(pitch_slider), p.readUserDebugParameter(yaw_slider)
            speed = p.readUserDebugParameter(speed_slider)

            start_pos = np.array([sx, sy, sz])
            end_pos = np.array([ex, ey, ez])
            orn = p.getQuaternionFromEuler([roll, pitch, yaw])

            current_params = (sx, sy, sz, ex, ey, ez, roll, pitch, yaw)

            if current_params != last_params and not running_trajectory:
                # Refresh visuals if the user moves sliders while IDLE
                p.removeAllUserDebugItems()
                
                # Draw the path using Pybullet Industrial
                # We use 10 samples so it doesn't draw a solid block of axes
                preview_path = pi.linear_interpolation(start_pos, end_pos, 10, start_orientation=orn, end_orientation=orn)
                preview_path.draw(pose=True, color=[0, 1, 0]) 
                
                last_params = current_params

            if current_exec > exec_clicks:
                exec_clicks = current_exec
                if not running_trajectory:
                    print("\n---> Calculating Inverse Kinematics Path...")
                    distance = np.linalg.norm(end_pos - start_pos)
                    samples = int((distance / speed) * 240.0)
                    if samples < 2:
                        samples = 2
                    
                    # Generate Dense ToolPath for movement
                    tool_path = pi.linear_interpolation(start_pos, end_pos, samples, start_orientation=orn, end_orientation=orn)
                    
                    safe_path_found = False
                    trajectory_points = []
                    
                    # Store current physical state before testing configurations
                    state_id = p.saveState()

                    # Disable async PyBullet GUI rendering to prevent visual flickering while we rapidly test joints!
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

                    for attempt, seed in enumerate(ik_seeds):
                        print(f"Testing configuration seed {attempt+1}/{len(ik_seeds)}...")
                        p.restoreState(state_id)
                        
                        # Reset robot to this potential seed
                        for i, j in enumerate(dof_joint_indices):
                            p.resetJointState(robot_id, j, seed[i])
                        
                        trajectory_points = []
                        collision = False
                        
                        for step_obj in tool_path:
                            # step_obj possesses (position, orientation, activation) tuples
                            t_pos = step_obj[0]
                            t_orn = step_obj[1]
                            
                            ik_sol = p.calculateInverseKinematics(
                                robot_id, tcp_link_index, t_pos, t_orn, 
                                maxNumIterations=100, residualThreshold=1e-4
                            )
                            
                            # Move simulation visually forward
                            for c, j in enumerate(dof_joint_indices):
                                p.resetJointState(robot_id, j, ik_sol[c])
                                
                            # Forward kinematics accuracy check: Did the IK actually reach the target, or is the path out of physical reach?
                            link_state = p.getLinkState(robot_id, tcp_link_index, computeForwardKinematics=True)
                            actual_pos = link_state[4] # link frame origin
                            
                            deviation = np.linalg.norm(np.array(actual_pos) - np.array(t_pos))
                            if deviation > tolerance_m:
                                print(f"     [!] Unreachable Path: IK deviation of {deviation*1000:.1f}mm detected (Limit: {args.tolerance}mm). Target is out of bounds or singular.")
                                collision = True
                                break
                            
                            if not collision_checker.is_collision_free():
                                print("     [!] Collision detected along path.")
                                collision = True
                                break
                            
                            trajectory_points.append(ik_sol)
                            
                        if not collision:
                            print("Safe collision-free path found!")
                            safe_path_found = True
                            break
                            
                    # Restore back to original physical state 
                    p.restoreState(state_id)
                    p.removeState(state_id)
                    
                    # Re-enable rendering now that the robot is physically restored to its safe position
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                    
                    if not safe_path_found:
                        print("ERROR: The path is not possible! Self collisions occurred in all attempts. Aborting.")
                        running_trajectory = False
                    else:
                        print("Executing trajectory... (Press Cancel to stop)")
                        
                        # To move smoothly to the start point, we interpolate joint positions from current state to the start state
                        current_joints = [p.getJointState(robot_id, j)[0] for j in dof_joint_indices]
                        transition_steps = 120 # Half a second
                        transition_traj = []
                        for t in range(transition_steps):
                            alpha = t / float(transition_steps)
                            # Cosine smoothing (ease-in ease-out)
                            blend = (1.0 - np.cos(alpha * np.pi)) / 2.0
                            interp_joints = [(1.0 - blend) * c + blend * n for c, n in zip(current_joints, trajectory_points[0])]
                            transition_traj.append(interp_joints)
                            
                        # Prepend the smoothing trajectory
                        trajectory_points = transition_traj + trajectory_points
                            
                        traj_idx = 0
                        running_trajectory = True

            if running_trajectory:
                if traj_idx < len(trajectory_points):
                    ik_sol = trajectory_points[traj_idx]
                    for i, joint_idx in enumerate(dof_joint_indices):
                        p.setJointMotorControl2(
                            bodyIndex=robot_id,
                            jointIndex=joint_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=ik_sol[i],
                            force=5000.0,
                            maxVelocity=5.0,
                            positionGain=0.05,
                            velocityGain=1.0
                        )
                    traj_idx += 1
                else:
                    print("--> Trajectory complete. Holding position.")
                    running_trajectory = False

            p.stepSimulation()
            time.sleep(1/240.)
            
    except KeyboardInterrupt:
        pass
        
    p.disconnect()

if __name__ == '__main__':
    main()
