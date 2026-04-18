import pybullet as p
import pybullet_data
import time
import math

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
boxId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

home_pose = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4, 0.04, 0.04]
for i in range(9):
    p.resetJointState(boxId, i, home_pose[i])

p.stepSimulation()
pts = p.getContactPoints(bodyA=boxId, bodyB=boxId)
for pt in pts:
    print(f"Self-collision between link {pt[3]} and {pt[4]}")
