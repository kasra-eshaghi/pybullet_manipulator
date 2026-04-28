import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
robot_id = p.loadURDF("welding_robot2.urdf", [0,0,0], useFixedBase=True)
for i in range(-1, p.getNumJoints(robot_id)):
    if i == -1:
        print(f"Link {i}: base_link")
    else:
        print(f"Link {i}: {p.getJointInfo(robot_id, i)[12].decode('utf-8')}")
