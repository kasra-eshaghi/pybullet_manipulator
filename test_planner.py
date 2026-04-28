import pybullet as p
import pybullet_data
from rrt_planner import RRTPlanner
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
boxId = p.loadURDF("welding_robot2.urdf", [0,0,0], useFixedBase=True)

reset_angles = [0, -1.57, 1.57, -1.57, -1.57, 0]
active_joints = []
for i in range(p.getNumJoints(boxId)):
    joint_info = p.getJointInfo(boxId, i)
    if joint_info[2] in [0, 1]: 
        active_joints.append(joint_info[0])
        p.resetJointState(boxId, i, reset_angles[len(active_joints)-1])

ee_link_id = None
for i in range(p.getNumJoints(boxId)):
    joint_info = p.getJointInfo(boxId, i)            
    if str(joint_info[12].decode('utf-8')) == "tcp_link":
        ee_link_id = joint_info[0]

ee_state = p.getLinkState(boxId, ee_link_id)
ee_pos = list(ee_state[0])
ee_euler = list(p.getEulerFromQuaternion(ee_state[1]))

start_pos = [ee_pos[0], ee_pos[1] - 0.2, ee_pos[2]]
goal_pos = [ee_pos[0], ee_pos[1] + 0.2, ee_pos[2]]

planner = RRTPlanner(
    urdf_path="welding_robot2.urdf", 
    ee_link_name="tcp_link",
    base_pos=[0,0,0], 
    base_orn=[0,0,0,1], 
    config_path="rrt_config.yaml"
)

paths = planner.plan_constrained_t_space(start_pos, ee_euler, goal_pos, ee_euler)
if paths:
    path = list(paths.values())[0]
    min_z = 1000
    for q in path:
        for i in range(p.getNumJoints(boxId)):
            if p.getJointInfo(boxId, i)[2] in [0,1]:
                p.resetJointState(boxId, i, q[active_joints.index(p.getJointInfo(boxId, i)[0])])
        
        for i in range(p.getNumJoints(boxId)):
            z = p.getLinkState(boxId, i)[0][2]
            if z < min_z and i > 1:
                min_z = z
    print("Lowest Z achieved by active links:", min_z)
