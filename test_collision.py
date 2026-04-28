import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("welding_robot2.urdf", [0,0,0], useFixedBase=True)
plane_id = p.loadURDF("plane.urdf", [0, 0, -0.01])

reset_angles = [0, -1.57, 1.57, -1.57, -1.57, 0]
active_joints = []
for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    if joint_info[2] in [0, 1]: 
        active_joints.append(joint_info[0])
        p.resetJointState(robot_id, i, reset_angles[len(active_joints)-1])

p.performCollisionDetection()
pts = p.getContactPoints(bodyA=robot_id, bodyB=plane_id)
print("Collision points with reset_angles:", len(pts))
if pts:
    print("Links colliding with plane:")
    for pt in pts:
        link_idx = pt[3]
        if link_idx == -1:
            print("- base_link")
        else:
            print(f"- {p.getJointInfo(robot_id, link_idx)[12].decode('utf-8')}")
