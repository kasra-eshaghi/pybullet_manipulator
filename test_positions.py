import pybullet as p
import pybullet_data
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
print("EE Pos:", ee_state[0])
print("EE Euler:", p.getEulerFromQuaternion(ee_state[1]))
