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
        
        t_hist = []
        qt_hist, qt_dot_hist, qt_ddot_hist = [], [], []
        qs_hist, qs_dot_hist, qs_ddot_hist = [], [], []
        s_hist = []
        
        while current_time <= traj_time:
            qt, qt_dot, qt_ddot, qs, qs_dot, qs_ddot, s_val = trajectory_eval(current_time)
            p.setJointMotorControlArray(
                boxId, active_joints, p.POSITION_CONTROL, 
                targetPositions=qt, targetVelocities=qt_dot,
                forces=np.ones(len(active_joints))*100
            )
            p.stepSimulation()
            
            ee_state = p.getLinkState(boxId, ee_link_id)
            fk_pos = np.array(ee_state[0])
            exec_points.append(fk_pos)
            
            t_hist.append(current_time)
            qt_hist.append(qt)
            qt_dot_hist.append(qt_dot)
            qt_ddot_hist.append(qt_ddot)
            qs_hist.append(qs)
            qs_dot_hist.append(qs_dot)
            qs_ddot_hist.append(qs_ddot)
            s_hist.append(s_val)
            
            if is_constrained:
                # Project FK position orthogonally onto constraint vector to find TRUE geometric s deviation
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
                    
        # Analytical Plotting Engine
        t_arr = np.array(t_hist)
        s_arr = np.array(s_hist)
        
        qt_arr = np.array(qt_hist)
        qt_dot_arr = np.array(qt_dot_hist)
        qt_ddot_arr = np.array(qt_ddot_hist)
        
        qs_arr = np.array(qs_hist)
        qs_dot_arr = np.array(qs_dot_hist)
        qs_ddot_arr = np.array(qs_ddot_hist)
        
        xt_arr = np.array(exec_points)
        
        if len(t_arr) > 1:
            # EE Derivatives 
            xt_dot = np.gradient(xt_arr, t_arr, axis=0) # velocity vs time
            xt_ddot = np.gradient(xt_dot, t_arr, axis=0) # accel vs time
            
            ds_dt = np.gradient(s_arr, t_arr)
            ds_dt_safe = np.where(np.abs(ds_dt) < 1e-6, 1e-6, ds_dt)
            
            xs_dot = xt_dot / ds_dt_safe[:, np.newaxis] # velocity vs s
            xs_ddot = np.gradient(xs_dot, t_arr, axis=0) / ds_dt_safe[:, np.newaxis] # accel vs s
            
            fig, axs = plt.subplots(6, 2, figsize=(16, 24))
            
            # Pre-calculate corresponding Time and S values for the actual topological waypoints for plotting demarcations
            node_s_vals = list(range(len(path)))
            dists = [planner.distance(path[i], path[i+1]) for i in range(len(path)-1)]
            tot_dist = sum(dists)
            node_t_vals = [0.0]
            if tot_dist > 1e-6:
                for d in dists:
                    node_t_vals.append(node_t_vals[-1] + traj_time * d / tot_dist)
            
            # --- Column 1: Time Domain ---
            # 1. Joint angles vs time
            for i in range(qt_arr.shape[1]):
                axs[0, 0].plot(t_arr, qt_arr[:, i], label=f'J{i}')
            axs[0, 0].set_ylabel('Angle (rad)')
            axs[0, 0].set_title('1. Joint Angles vs Time')
            
            # 2. Joint velocity vs time
            for i in range(qt_dot_arr.shape[1]):
                axs[1, 0].plot(t_arr, qt_dot_arr[:, i], label=f'J{i}_dot')
            axs[1, 0].set_ylabel('Velocity (rad/s)')
            axs[1, 0].set_title('2. Joint Velocities vs Time')
            
            # 3. Joint acceleration vs time
            for i in range(qt_ddot_arr.shape[1]):
                axs[2, 0].plot(t_arr, qt_ddot_arr[:, i], label=f'J{i}_ddot')
            axs[2, 0].set_ylabel('Accel (rad/s^2)')
            axs[2, 0].set_title('3. Joint Accelerations vs Time')
            
            # 4. EE pos vs time
            axs[3, 0].plot(t_arr, xt_arr[:, 0], label='X')
            axs[3, 0].plot(t_arr, xt_arr[:, 1], label='Y')
            axs[3, 0].plot(t_arr, xt_arr[:, 2], label='Z')
            axs[3, 0].set_ylabel('Position (m)')
            axs[3, 0].set_title('4. EE Position vs Time')
            
            # 5. EE vel vs time
            axs[4, 0].plot(t_arr, xt_dot[:, 0], label='X_dot')
            axs[4, 0].plot(t_arr, xt_dot[:, 1], label='Y_dot')
            axs[4, 0].plot(t_arr, xt_dot[:, 2], label='Z_dot')
            axs[4, 0].set_ylabel('Velocity (m/s)')
            axs[4, 0].set_title('5. EE Velocity vs Time')
            
            # 6. EE accel vs time
            axs[5, 0].plot(t_arr, xt_ddot[:, 0], label='X_ddot')
            axs[5, 0].plot(t_arr, xt_ddot[:, 1], label='Y_ddot')
            axs[5, 0].plot(t_arr, xt_ddot[:, 2], label='Z_ddot')
            axs[5, 0].set_ylabel('Accel (m/s^2)')
            axs[5, 0].set_title('6. EE Acceleration vs Time')
            
            # --- Column 2: S-Space Domain ---
            # 7. Joint angles vs s
            for i in range(qs_arr.shape[1]):
                axs[0, 1].plot(s_arr, qs_arr[:, i], label=f'J{i}')
            axs[0, 1].set_ylabel('Angle (rad)')
            axs[0, 1].set_title('7. Joint Angles vs s')
            
            # 8. Joint velocity vs s 
            for i in range(qs_dot_arr.shape[1]):
                axs[1, 1].plot(s_arr, qs_dot_arr[:, i], label=f'J{i}_dot')
            axs[1, 1].set_ylabel('Velocity (dq/ds)')
            axs[1, 1].set_title('8. Joint Velocities vs s')
            
            # 9. Joint acceleration vs s
            for i in range(qs_ddot_arr.shape[1]):
                axs[2, 1].plot(s_arr, qs_ddot_arr[:, i], label=f'J{i}_ddot')
            axs[2, 1].set_ylabel('Accel (d2q/ds2)')
            axs[2, 1].set_title('9. Joint Accelerations vs s')
            
            # 10. EE pos vs s
            axs[3, 1].plot(s_arr, xt_arr[:, 0], label='X')
            axs[3, 1].plot(s_arr, xt_arr[:, 1], label='Y')
            axs[3, 1].plot(s_arr, xt_arr[:, 2], label='Z')
            axs[3, 1].set_ylabel('Position (m)')
            axs[3, 1].set_title('10. EE Position vs s')
            
            # 11. EE vel vs s
            axs[4, 1].plot(s_arr, xs_dot[:, 0], label='X_dot')
            axs[4, 1].plot(s_arr, xs_dot[:, 1], label='Y_dot')
            axs[4, 1].plot(s_arr, xs_dot[:, 2], label='Z_dot')
            axs[4, 1].set_ylabel('Velocity (dx/ds)')
            axs[4, 1].set_title('11. EE Velocity vs s')
            
            # 12. EE accel vs s
            axs[5, 1].plot(s_arr, xs_ddot[:, 0], label='X_ddot')
            axs[5, 1].plot(s_arr, xs_ddot[:, 1], label='Y_ddot')
            axs[5, 1].plot(s_arr, xs_ddot[:, 2], label='Z_ddot')
            axs[5, 1].set_ylabel('Accel (d2x/ds2)')
            axs[5, 1].set_title('12. EE Acceleration vs s')
            
            # Formatting and Legends
            for row in range(6):
                for col in range(2):
                    n_vals = node_t_vals if col == 0 else node_s_vals
                    for nv in n_vals:
                        axs[row, col].axvline(x=nv, color='gray', linestyle='--', alpha=0.5)
                        
                    if col == 0:
                        axs[row, col].set_xlim(-0.05 * traj_time, 1.05 * traj_time)
                    else:
                        axs[row, col].set_xlim(-0.05, len(path) - 1 + 0.05)
                        
                    axs[row, col].legend(loc='upper right', fontsize='x-small', ncol=3)
                    
                    if row < 5:
                        axs[row, col].set_xticklabels([])
                    else:
                        axs[row, col].set_xlabel('Time (s)' if col == 0 else 'Task Space Fraction (s)')
                        
            plt.subplots_adjust(hspace=0.4)
            plt.tight_layout(h_pad=1.5)
            plt.show(block=False)

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
