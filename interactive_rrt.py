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
    planeId = p.loadURDF("plane.urdf")
    
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    boxId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

    active_joints = []
    active_joint_names = []
    limits = []
    ee_link_name = "panda_grasptarget"
    ee_link_id = None

    for i in range(p.getNumJoints(boxId)):
        joint_info = p.getJointInfo(boxId, i)
        joint_type = joint_info[2]
        if joint_type == 0 or joint_type == 1: 
            active_joints.append(joint_info[0])
            active_joint_names.append(joint_info[1])
            limits.append(joint_info[8:10])
            
        if str(joint_info[12].decode('utf-8')) == ee_link_name:
            ee_link_id = joint_info[0]

    for i in range(len(active_joints)):
        ll, ul = limits[i]
        if ll >= ul:
            ll, ul = -3.14159, 3.14159
        mid_val = (ll + ul) / 2.0
        p.resetJointState(boxId, active_joints[i], mid_val)

    p.stepSimulation()

    joint_states = p.getJointStates(boxId, active_joints)
    qi = np.array([state[0] for state in joint_states])
    
    if ee_link_id is not None:
        ee_state = p.getLinkState(boxId, ee_link_id)
        ee_pos = ee_state[0]
        ee_euler = p.getEulerFromQuaternion(ee_state[1])
    else:
        ee_pos = [0, 0, 0]
        ee_euler = [0, 0, 0]

    # --- Construct All UI Sliders ---
    c_sliders = []
    for i in range(len(active_joints)):
        joint_name = active_joint_names[i].decode('utf-8')
        lower_lim, upper_lim = limits[i]
        if lower_lim >= upper_lim:
            lower_lim, upper_lim = -3.14159, 3.14159
        c_sliders.append(p.addUserDebugParameter(f"C-Goal: {joint_name}", lower_lim, upper_lim, qi[i]))
    
    btn_c_go = p.addUserDebugParameter("Plan & Go (C-Space)!", 1, 0, 0)
    
    ts_labels = ["T-Goal X", "T-Goal Y", "T-Goal Z", "T-Goal Roll", "T-Goal Pitch", "T-Goal Yaw"]
    pos_eul = list(ee_pos) + list(ee_euler)
    ts_bounds = [
        (-1.0, 1.0, pos_eul[0]), (-1.0, 1.0, pos_eul[1]), ( 0.0, 1.5, pos_eul[2]),
        (-3.14159, 3.14159, pos_eul[3]), (-3.14159, 3.14159, pos_eul[4]), (-3.14159, 3.14159, pos_eul[5])
    ]
    ts_sliders = []
    for label, b in zip(ts_labels, ts_bounds):
        ts_sliders.append(p.addUserDebugParameter(label, b[0], b[1], b[2]))
        
    btn_ts_go = p.addUserDebugParameter("Plan & Go (T-Space)!", 1, 0, 0)
    
    obs_pos_sliders = [
        p.addUserDebugParameter("Obs Pos X", -1.0, 1.0, 0.5),
        p.addUserDebugParameter("Obs Pos Y", -1.0, 1.0, 0.0),
        p.addUserDebugParameter("Obs Pos Z", 0.0, 1.5, 0.5)
    ]
    obs_dim_sliders = [
        p.addUserDebugParameter("Obs Half-Length (X)", 0.01, 1.0, 0.1),
        p.addUserDebugParameter("Obs Half-Width (Y)", 0.01, 1.0, 0.1),
        p.addUserDebugParameter("Obs Half-Height (Z)", 0.01, 1.0, 0.1)
    ]

    prev_btn_c_val = p.readUserDebugParameter(btn_c_go)
    prev_btn_ts_val = p.readUserDebugParameter(btn_ts_go)

    # Obstacle Setup in GUI Server
    gui_obs_id = None
    gui_obs_col = None
    gui_obs_vis = None
    last_obs_state = None

    def update_gui_obstacle():
        nonlocal gui_obs_id, gui_obs_col, gui_obs_vis, last_obs_state
        op = [p.readUserDebugParameter(s) for s in obs_pos_sliders]
        od = [p.readUserDebugParameter(s) for s in obs_dim_sliders]
        current_state = op + od
        
        # Only rebuild if values actually changed to avoid flickering
        if last_obs_state != current_state:
            if gui_obs_id is not None:
                p.removeBody(gui_obs_id)
            gui_obs_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=od)
            gui_obs_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=od, rgbaColor=[1, 0.5, 0, 0.7])
            gui_obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=gui_obs_col, baseVisualShapeIndex=gui_obs_vis, basePosition=op)
            last_obs_state = current_state

    update_gui_obstacle()
    
    planner = RRTPlanner(urdf_path="franka_panda/panda.urdf", active_joints=active_joints, config_path="rrt_config.yaml")

    print("Welcome! Use the GUI widgets to set goals and obstacles.")
    qs = None
    step_counter = 0
    line_x_id = -1
    line_y_id = -1
    line_z_id = -1
    tree_debug_items = []
    
    try:
        while True:
            c_go_val = p.readUserDebugParameter(btn_c_go)
            ts_go_val = p.readUserDebugParameter(btn_ts_go)
            
            trigger_plan = False
            qf = None
            
            if c_go_val > prev_btn_c_val:
                prev_btn_c_val = c_go_val
                qf = np.array([p.readUserDebugParameter(sid) for sid in c_sliders])
                print("Triggered C-Space Plan")
                trigger_plan = True
                
            elif ts_go_val > prev_btn_ts_val:
                prev_btn_ts_val = ts_go_val
                if ee_link_id is None:
                    print("Cannot plan in T-space, EE link not found.")
                else:
                    t_pos = [p.readUserDebugParameter(sid) for sid in ts_sliders[:3]]
                    t_euler = [p.readUserDebugParameter(sid) for sid in ts_sliders[3:]]
                    movable_joints = []
                    ll_list = []
                    ul_list = []
                    jr_list = []
                    rp_list = []
                    for i in range(p.getNumJoints(boxId)):
                        info = p.getJointInfo(boxId, i)
                        if info[2] != p.JOINT_FIXED:
                            movable_joints.append(i)
                            ll, ul = info[8], info[9]
                            if ll >= ul:
                                ll, ul = -3.14159, 3.14159
                            ll_list.append(ll)
                            ul_list.append(ul)
                            jr_list.append(ul - ll)
                            rp_list.append((ll + ul) / 2.0)
                            
                    ik_sol = p.calculateInverseKinematics(
                        boxId, ee_link_id, t_pos, p.getQuaternionFromEuler(t_euler),
                        lowerLimits=ll_list,
                        upperLimits=ul_list,
                        jointRanges=jr_list,
                        restPoses=rp_list,
                        maxNumIterations=100,
                        residualThreshold=1e-5
                    )
                    
                    qf = np.zeros(len(active_joints))
                    for i, joint_idx in enumerate(active_joints):
                        if joint_idx in movable_joints:
                            ik_idx = movable_joints.index(joint_idx)
                            qf[i] = ik_sol[ik_idx]
                            
                    print(f"Triggered T-Space Plan. IK solved joint configuration: {qf}")
                    trigger_plan = True
                
            if trigger_plan and qf is not None:
                op = [p.readUserDebugParameter(s) for s in obs_pos_sliders]
                od = [p.readUserDebugParameter(s) for s in obs_dim_sliders]
                planner.update_dynamic_obstacle(op, od)
                
                joint_states = p.getJointStates(boxId, active_joints)
                qi = np.array([state[0] for state in joint_states])
                
                # Clear previous tree visualization
                for item in tree_debug_items:
                    p.removeUserDebugItem(item)
                tree_debug_items.clear()
                
                print("Planning RRT* path...")
                path = planner.plan(qi, qf)
                
                # Visualize all raw sampled points regardless of success
                pts, _ = planner.get_tree_cartesian_nodes(ee_link_id)
                if pts:
                    batch_size = 500
                    for i in range(0, len(pts), batch_size):
                        chunk = pts[i:i+batch_size]
                        colors = [[0, 1, 1] for _ in chunk] # Cyan points for RRT tree
                        item_id = p.addUserDebugPoints(chunk, colors, pointSize=3.0)
                        tree_debug_items.append(item_id)
                
                if path:
                    traj_time = 5.0 
                    trajectory_eval = planner.generate_trajectory(path, traj_time)
                    
                    print("Executing trajectory...")
                    current_time = 0.0
                    qs_hist = np.empty((0, len(qi)))
                    
                    while current_time <= traj_time:
                        q_t, q_dot_t, _ = trajectory_eval(current_time)
                        p.setJointMotorControlArray(
                            boxId, active_joints, p.POSITION_CONTROL, 
                            targetPositions=q_t, targetVelocities=q_dot_t,
                            forces=np.ones(len(active_joints))*100
                        )
                        p.stepSimulation()
                        time.sleep(1./240.)
                        current_time += 1./240.

                        joint_states = p.getJointStates(boxId, active_joints)
                        qs_hist = np.vstack((qs_hist, np.array([state[0] for state in joint_states])))
                        
                    print("Trajectory completed!")
                    qs = qs_hist
                    
                    # Hard-reset the previous button values to prevent multi-trigger queues 
                    # from sliders manipulated DURING execution
                    prev_btn_c_val = p.readUserDebugParameter(btn_c_go)
                    prev_btn_ts_val = p.readUserDebugParameter(btn_ts_go)
                else:
                    print("Could not find a valid path to the goal configuration.")
                    prev_btn_c_val = p.readUserDebugParameter(btn_c_go)
                    prev_btn_ts_val = p.readUserDebugParameter(btn_ts_go)
                    
            # T-Space Frame Visualizer continuously updates
            if step_counter % 8 == 0 and ee_link_id is not None:
                t_pos = [p.readUserDebugParameter(sid) for sid in ts_sliders[:3]]
                t_euler = [p.readUserDebugParameter(sid) for sid in ts_sliders[3:]]
                target_quat = p.getQuaternionFromEuler(t_euler)
                
                axis_len = 0.1
                x_end, _ = p.multiplyTransforms(t_pos, target_quat, [axis_len, 0, 0], [0, 0, 0, 1])
                y_end, _ = p.multiplyTransforms(t_pos, target_quat, [0, axis_len, 0], [0, 0, 0, 1])
                z_end, _ = p.multiplyTransforms(t_pos, target_quat, [0, 0, axis_len], [0, 0, 0, 1])
                
                if line_x_id < 0:
                    line_x_id = p.addUserDebugLine(t_pos, x_end, lineColorRGB=[1, 0, 0], lineWidth=2)
                    line_y_id = p.addUserDebugLine(t_pos, y_end, lineColorRGB=[0, 1, 0], lineWidth=2)
                    line_z_id = p.addUserDebugLine(t_pos, z_end, lineColorRGB=[0, 0, 1], lineWidth=2)
                else:
                    p.addUserDebugLine(t_pos, x_end, lineColorRGB=[1, 0, 0], lineWidth=2, replaceItemUniqueId=line_x_id)
                    p.addUserDebugLine(t_pos, y_end, lineColorRGB=[0, 1, 0], lineWidth=2, replaceItemUniqueId=line_y_id)
                    p.addUserDebugLine(t_pos, z_end, lineColorRGB=[0, 0, 1], lineWidth=2, replaceItemUniqueId=line_z_id)
                    
            # Check Obstacle Changes
            if step_counter % 15 == 0:
                update_gui_obstacle()

            p.stepSimulation()
            time.sleep(1./240.)
            step_counter += 1
                
    except KeyboardInterrupt:
        pass
    p.disconnect()

    if qs is not None:
        plt.plot(qs)
        plt.title("RRT* Joint Trajectories (Last Run)")
        plt.xlabel("Simulation Timesteps")
        plt.ylabel("Joint Angles (rad)")
        plt.show()

if __name__ == "__main__":
    main()
