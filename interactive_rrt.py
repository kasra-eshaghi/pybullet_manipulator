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
    limits = []
    for i in range(p.getNumJoints(boxId)):
        joint_info = p.getJointInfo(boxId, i)
        joint_type = joint_info[2]
        if joint_type == 0 or joint_type == 1: 
            active_joints.append(joint_info[0])
            ll, ul = joint_info[8], joint_info[9]
            if ll >= ul:
                ll, ul = -3.14159, 3.14159
            limits.append((ll, ul))
            mid_val = (ll + ul) / 2.0
            p.resetJointState(boxId, i, mid_val)
    p.stepSimulation()

    # get state of ee_link
    ee_state = p.getLinkState(boxId, ee_link_id)
    ee_pos = ee_state[0]
    ee_euler = p.getEulerFromQuaternion(ee_state[1])

    # --- Construct All UI Sliders ---
    # goal position sliders
    ts_info = [
        ("T-Goal X", -1.0, 1.0, ee_pos[0]), 
        ("T-Goal Y", -1.0, 1.0, ee_pos[1]), 
        ("T-Goal Z", 0.0, 1.5, ee_pos[2]), 
        ("T-Goal Roll", -3.14159, 3.14159, ee_euler[0]), 
        ("T-Goal Pitch", -3.14159, 3.14159, ee_euler[1]), 
        ("T-Goal Yaw", -3.14159, 3.14159, ee_euler[2])
    ]
    ts_sliders = []
    for (label, ll, ul, val) in ts_info:
        ts_sliders.append(p.addUserDebugParameter(label, ll, ul, val))
        
    btn_ts_go = p.addUserDebugParameter("Plan & Go!", 1, 0, 0)
    prev_btn_ts_val = p.readUserDebugParameter(btn_ts_go)
    
    # obstacle position sliders
    obs_info = [
        ("Obs Pos X", -1.0, 1.0, 0.5),
        ("Obs Pos Y", -1.0, 1.0, 0.0),
        ("Obs Pos Z", 0.0, 1.5, 0.5),
        ("Obs Half-Length (X)", 0.01, 1.0, 0.1),
        ("Obs Half-Width (Y)", 0.01, 1.0, 0.1),
        ("Obs Half-Height (Z)", 0.01, 1.0, 0.1)
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
        
        # Only rebuild if values actually changed to avoid flickering
        if last_obs_state != current_state:
            if gui_obs_id is not None:
                p.removeBody(gui_obs_id)
            gui_obs_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=od)
            gui_obs_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=od, rgbaColor=[1, 0.5, 0, 0.7])
            gui_obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=gui_obs_col, baseVisualShapeIndex=gui_obs_vis, basePosition=op)
            last_obs_state = current_state
    update_gui_obstacle()
    
    # create path planner
    planner = RRTPlanner(
        urdf_path="franka_panda/panda.urdf", 
        ee_link_name="panda_grasptarget",
        base_pos=base_pos, 
        base_orn=base_orn, 
        config_path="rrt_config.yaml"
    )

    print("Welcome! Use the GUI widgets to set goals and obstacles.")
    qs = None
    step_counter = 0
    line_x_id = -1
    line_y_id = -1
    line_z_id = -1
    tree_debug_items = []
    
    try:
        while True:
            ts_go_val = p.readUserDebugParameter(btn_ts_go)
                      
            if ts_go_val > prev_btn_ts_val:
                prev_btn_ts_val = ts_go_val
                t_pos = [p.readUserDebugParameter(sid) for sid in ts_sliders[:3]]
                t_euler = [p.readUserDebugParameter(sid) for sid in ts_sliders[3:]]
                
                print(f"Triggered planning")
                
                op = [p.readUserDebugParameter(s) for s in obs_sliders[:3]]
                od = [p.readUserDebugParameter(s) for s in obs_sliders[3:]]
                planner.update_dynamic_obstacle(op, od)
                
                joint_states = p.getJointStates(boxId, active_joints)
                qi = np.array([state[0] for state in joint_states])
                
                # Clear previous tree visualization
                for item in tree_debug_items:
                    p.removeUserDebugItem(item)
                tree_debug_items.clear()
                
                print("Planning RRT* path...")
                path = planner.plan_t_space(qi, t_pos, t_euler)
                
                # Visualize all raw sampled points regardless of success
                pts, _ = planner.get_tree_cartesian_nodes()
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
                    prev_btn_ts_val = p.readUserDebugParameter(btn_ts_go)
                else:
                    print("Could not find a valid path to the goal configuration.")
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


if __name__ == "__main__":
    main()
