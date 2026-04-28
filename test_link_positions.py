import pybullet as p
import pybullet_data
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("welding_robot2.urdf", [0,0,0], useFixedBase=True)

reset_angles = [0, -1.57, 1.57, -1.57, -1.57, 0]
active_joints = []
for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    if joint_info[2] in [0, 1]: 
        active_joints.append(joint_info[0])
        p.resetJointState(robot_id, i, reset_angles[len(active_joints)-1])

for i in range(p.getNumJoints(robot_id)):
    state = p.getLinkState(robot_id, i)
    name = p.getJointInfo(robot_id, i)[12].decode('utf-8')
    print(f"{name}: Z = {state[0][2]:.3f}")
