import pybullet as p
import pybullet_data
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
boxId = p.loadURDF("welding_robot2.urdf", [0,0,0], useFixedBase=True)
count = 0
for i in range(p.getNumJoints(boxId)):
    joint_info = p.getJointInfo(boxId, i)
    if joint_info[2] in [0, 1]:
        count += 1
print("Num active joints:", count)
