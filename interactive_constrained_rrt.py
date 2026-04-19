import numpy as np
import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
from rrt_planner import RRTPlanner

def main():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    # create plane and robot
    planeId = p.loadURDF("plane.urdf")
    base_pos = [0, 0, 0]
    base_orn = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("franka_panda/panda.urdf", base_pos, base_orn, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

    # find ee_link name 
    ee_link_name = "panda_grasptarget"
    ee_link_id = None
    for i in range(p.getNumJoints(boxId)):
        joint_info = p.getJointInfo(boxId, i)            
        if str(joint_info[12].decode('utf-8')) == ee_link_name:
            ee_link_id = joint_info[0]
    if ee_link_id is None:
        raise ValueError("EE link not found")

    # reset position of robot
    active_joints = []
    for i in range(p.getNumJoints(boxId)):
        joint_info = p.getJointInfo(boxId, i)
        joint_type = joint_info[2]
        if joint_type == 0 or joint_type == 1: 
            active_joints.append(joint_info[0])
            ll, ul = joint_info[8], joint_info[9]
            if ll >= ul:
                ll, ul = -3.14159, 3.14159
            mid_val = (ll + ul) / 2.0
            p.resetJointState(boxId, i, mid_val)
    p.stepSimulation()

    # get state of ee_link
    ee_state = p.getLinkState(boxId, ee_link_id)
    ee_pos = ee_state[0]
    ee_euler = p.getEulerFromQuaternion(ee_state[1])

    # --- Construct All UI Sliders ---
    start_info = [
        ("Start X", -1.0, 1.0, ee_pos[0]), 
        ("Start Y", -1.0, 1.0, ee_pos[1] - 0.2), 
        ("Start Z", 0.0, 1.5, ee_pos[2]), 
        ("Start Roll", -3.14159, 3.14159, ee_euler[0]), 
        ("Start Pitch", -3.14159, 3.14159, ee_euler[1]), 
        ("Start Yaw", -3.14159, 3.14159, ee_euler[2])
    ]
    start_sliders = []
    for (label, ll, ul, val) in start_info:
        start_sliders.append(p.addUserDebugParameter(label, ll, ul, val))

    goal_info = [
        ("Goal X", -1.0, 1.0, ee_pos[0]), 
        ("Goal Y", -1.0, 1.0, ee_pos[1] + 0.2), 
        ("Goal Z", 0.0, 1.5, ee_pos[2]), 
        ("Goal Roll", -3.14159, 3.14159, ee_euler[0]), 
        ("Goal Pitch", -3.14159, 3.14159, ee_euler[1]), 
        ("Goal Yaw", -3.14159, 3.14159, ee_euler[2])
    ]
    goal_sliders = []
    for (label, ll, ul, val) in goal_info:
        goal_sliders.append(p.addUserDebugParameter(label, ll, ul, val))
        
    btn_go = p.addUserDebugParameter("Plan Constrained & Go!", 1, 0, 0)
    prev_btn_val = p.readUserDebugParameter(btn_go)
    
    obs_info = [
        ("Obs Pos X", -1.0, 1.0, -1.5),
        ("Obs Pos Y", -1.0, 1.0, 0.0),
        ("Obs Pos Z", 0.0, 1.5, 0.5),
        ("Obs Half-Length", 0.01, 1.0, 0.1),
        ("Obs Half-Width", 0.01, 1.0, 0.1),
        ("Obs Half-Height", 0.01, 1.0, 0.1)
    ]
    obs_sliders = []
    for (label, ll, ul, val) in obs_info:
        obs_sliders.append(p.addUserDebugParameter(label, ll, ul, val))

    # Obstacle Setup in GUI Server
    gui_obs_id = None
    gui_obs_col = None
    gui_obs_vis = None
    last_obs_state = None

    def update_gui_obstacle():
        nonlocal gui_obs_id, gui_obs_col, gui_obs_vis, last_obs_state
        op = [p.readUserDebugParameter(s) for s in obs_sliders[:3]]
        od = [p.readUserDebugParameter(s) for s in obs_sliders[3:]]
        current_state = op + od
        if last_obs_state != current_state:
            if gui_obs_id is not None:
                p.removeBody(gui_obs_id)
            gui_obs_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=od)
            gui_obs_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=od, rgbaColor=[1, 0.5, 0, 0.7])
            gui_obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=gui_obs_col, baseVisualShapeIndex=gui_obs_vis, basePosition=op)
            last_obs_state = current_state
    update_gui_obstacle()
    
    planner = RRTPlanner(
        urdf_path="franka_panda/panda.urdf", 
        ee_link_name="panda_grasptarget",
        base_pos=base_pos, 
        base_orn=base_orn, 
        config_path="rrt_config.yaml"
    )

    print("Constrained RRT GUI Loaded! Adjust start and goal nodes.")
    
    step_counter = 0
    frame_debug_ids = {}   # Dict tracking drawing IDs for dynamic updates
    connecting_line_id = -1
    trajectory_line_ids = []
    
    def execute_trajectory(path, traj_time=5.0, is_constrained=False, s_pos=None, g_pos=None):
        trajectory_eval = planner.generate_trajectory(path, traj_time)
        current_time = 0.0
        
        max_dist = 0.0
        exec_points = []
        
        while current_time <= traj_time:
            q_t, q_dot_t, q_ddot_t = trajectory_eval(current_time)
            p.setJointMotorControlArray(
                boxId, active_joints, p.POSITION_CONTROL, 
                targetPositions=q_t, targetVelocities=q_dot_t,
                forces=np.ones(len(active_joints))*100
            )
            p.stepSimulation()
            
            if is_constrained:
                ee_state = p.getLinkState(boxId, ee_link_id)
                fk_pos = np.array(ee_state[0])
                exec_points.append(fk_pos)
                
                # Project FK position orthogonally onto constraint vector to find TRUE geometric s
                vec_a = np.array(s_pos)
                vec_b = np.array(g_pos)
                line_vec = vec_b - vec_a
                sq_len = np.dot(line_vec, line_vec)
                
                if sq_len > 1e-8:
                    s_true = np.dot(fk_pos - vec_a, line_vec) / sq_len
                    s_true = float(np.clip(s_true, 0.0, 1.0))
                else:
                    s_true = 0.0
                
                target_pos = vec_a + s_true * line_vec
                
                dist = np.linalg.norm(fk_pos - target_pos)
                if dist > max_dist:
                    max_dist = dist
                    
            time.sleep(1./240.)
            current_time += 1./240.
            
        if is_constrained:
            print(f"Max trajectory physics deviation across constraint vector: {max_dist:.5f} meters")
            
            # Subsample lines to avoid PyBullet buffer overflow (causing User Debug Draw Failed / X11 Crash)
            step_draw = max(1, len(exec_points) // 200)
            if len(exec_points) > 1:
                for i in range(step_draw, len(exec_points), step_draw):
                    lid = p.addUserDebugLine(exec_points[i-step_draw].tolist(), exec_points[i].tolist(), lineColorRGB=[1, 0, 1], lineWidth=4)
                    trajectory_line_ids.append(lid)
                    

    try:
        while True:
            go_val = p.readUserDebugParameter(btn_go)
                      
            if go_val > prev_btn_val:
                prev_btn_val = go_val
                
                # Clear previous trajectory lines
                for lid in trajectory_line_ids:
                    p.removeUserDebugItem(lid)
                trajectory_line_ids.clear()
                
                s_pos = [p.readUserDebugParameter(sid) for sid in start_sliders[:3]]
                s_euler = [p.readUserDebugParameter(sid) for sid in start_sliders[3:]]
                g_pos = [p.readUserDebugParameter(sid) for sid in goal_sliders[:3]]
                g_euler = [p.readUserDebugParameter(sid) for sid in goal_sliders[3:]]
                
                op = [p.readUserDebugParameter(s) for s in obs_sliders[:3]]
                od = [p.readUserDebugParameter(s) for s in obs_sliders[3:]]
                planner.update_dynamic_obstacle(op, od)
                
                joint_states = p.getJointStates(boxId, active_joints)
                qi = np.array([state[0] for state in joint_states])
                
                print("\n --- Stage 2: Backwards Propagated Constrained Planning ---")
                valid_paths_dict = planner.plan_constrained_t_space(s_pos, s_euler, g_pos, g_euler)
                
                if not valid_paths_dict:
                    print("Constrained planner could not discover any valid structural pathways. Aborting pipeline.")
                else:
                    print(f"Stage 2 produced {len(valid_paths_dict)} unique safe start roots!")
                    valid_start_qs = [np.array(sq) for sq in valid_paths_dict.keys()]
                    
                    print("\n --- Stage 1: Transit to optimal Start Pose root ---")
                    start_path = planner.plan(qi, valid_start_qs)
                    
                    if start_path:
                        print("Standard RRT transit successful. Executing transit path...")
                        execute_trajectory(start_path)
                        
                        end_q = start_path[-1]
                        best_match = None
                        best_dist = float('inf')
                        for key_tuple in valid_paths_dict.keys():
                            d = np.linalg.norm(np.array(key_tuple) - end_q)
                            if d < best_dist:
                                best_dist = d
                                best_match = key_tuple
                                
                        if best_dist < 1e-4:
                            c_path = valid_paths_dict[best_match]
                            print(f"\n --- Stage 2: Constrained path execution beginning! ({len(c_path)} nodes) ---")
                            execute_trajectory(c_path, is_constrained=True, s_pos=s_pos, g_pos=g_pos)
                            print("Full Pipeline Execution achieved gracefully.")
                        else:
                            print("CRITICAL LOGIC ERROR: Transit path reached a node not registered in Stage 2 mapping!")
                    else:
                        print("No path found from start to any of the valid start roots. Try again.")
                    
                prev_btn_val = p.readUserDebugParameter(btn_go)
                
            # Interactively draw frames
            if step_counter % 8 == 0:
                s_pos = [p.readUserDebugParameter(sid) for sid in start_sliders[:3]]
                s_euler = [p.readUserDebugParameter(sid) for sid in start_sliders[3:]]
                g_pos = [p.readUserDebugParameter(sid) for sid in goal_sliders[:3]]
                g_euler = [p.readUserDebugParameter(sid) for sid in goal_sliders[3:]]
                
                s_quat = p.getQuaternionFromEuler(s_euler)
                g_quat = p.getQuaternionFromEuler(g_euler)
                
                axis_len = 0.1
                
                # Draw Start Frame
                sx_end, _ = p.multiplyTransforms(s_pos, s_quat, [axis_len, 0, 0], [0, 0, 0, 1])
                sy_end, _ = p.multiplyTransforms(s_pos, s_quat, [0, axis_len, 0], [0, 0, 0, 1])
                sz_end, _ = p.multiplyTransforms(s_pos, s_quat, [0, 0, axis_len], [0, 0, 0, 1])
                
                # Draw Goal Frame
                gx_end, _ = p.multiplyTransforms(g_pos, g_quat, [axis_len, 0, 0], [0, 0, 0, 1])
                gy_end, _ = p.multiplyTransforms(g_pos, g_quat, [0, axis_len, 0], [0, 0, 0, 1])
                gz_end, _ = p.multiplyTransforms(g_pos, g_quat, [0, 0, axis_len], [0, 0, 0, 1])
                
                if 'sx' not in frame_debug_ids:
                    frame_debug_ids['sx'] = p.addUserDebugLine(s_pos, sx_end, lineColorRGB=[1, 0, 0], lineWidth=2)
                    frame_debug_ids['sy'] = p.addUserDebugLine(s_pos, sy_end, lineColorRGB=[0, 1, 0], lineWidth=2)
                    frame_debug_ids['sz'] = p.addUserDebugLine(s_pos, sz_end, lineColorRGB=[0, 0, 1], lineWidth=2)
                    
                    frame_debug_ids['gx'] = p.addUserDebugLine(g_pos, gx_end, lineColorRGB=[1, 0, 0], lineWidth=2)
                    frame_debug_ids['gy'] = p.addUserDebugLine(g_pos, gy_end, lineColorRGB=[0, 1, 0], lineWidth=2)
                    frame_debug_ids['gz'] = p.addUserDebugLine(g_pos, gz_end, lineColorRGB=[0, 0, 1], lineWidth=2)
                    
                    connecting_line_id = p.addUserDebugLine(s_pos, g_pos, lineColorRGB=[1, 1, 0], lineWidth=2)
                else:
                    p.addUserDebugLine(s_pos, sx_end, lineColorRGB=[1, 0, 0], lineWidth=2, replaceItemUniqueId=frame_debug_ids['sx'])
                    p.addUserDebugLine(s_pos, sy_end, lineColorRGB=[0, 1, 0], lineWidth=2, replaceItemUniqueId=frame_debug_ids['sy'])
                    p.addUserDebugLine(s_pos, sz_end, lineColorRGB=[0, 0, 1], lineWidth=2, replaceItemUniqueId=frame_debug_ids['sz'])
                    
                    p.addUserDebugLine(g_pos, gx_end, lineColorRGB=[1, 0, 0], lineWidth=2, replaceItemUniqueId=frame_debug_ids['gx'])
                    p.addUserDebugLine(g_pos, gy_end, lineColorRGB=[0, 1, 0], lineWidth=2, replaceItemUniqueId=frame_debug_ids['gy'])
                    p.addUserDebugLine(g_pos, gz_end, lineColorRGB=[0, 0, 1], lineWidth=2, replaceItemUniqueId=frame_debug_ids['gz'])
                    
                    p.addUserDebugLine(s_pos, g_pos, lineColorRGB=[1, 1, 0], lineWidth=2, replaceItemUniqueId=connecting_line_id)
                    
            if step_counter % 15 == 0:
                update_gui_obstacle()

            p.stepSimulation()
            time.sleep(1./240.)
            plt.pause(0.001)
            step_counter += 1
                
    except KeyboardInterrupt:
        pass
    p.disconnect()

if __name__ == "__main__":
    main()
