import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
print(pybullet_data.getDataPath())
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("franka_panda/panda.urdf",startPos, startOrientation, useFixedBase=True)

active_joints = []
active_joint_names = []
limits = []
ee_link_name = "panda_grasptarget"
ee_link_id = None

joint_types = []

print(f"number of joints: {p.getNumJoints(boxId)}")
for i in range(p.getNumJoints(boxId)):
    joint_info = p.getJointInfo(boxId, i)
    joint_type = joint_info[2]
    if joint_type == 0 or joint_type == 1:
        active_joints.append(joint_info[0])
        active_joint_names.append(joint_info[1])
        limits.append(joint_info[8:10])
        joint_types.append(joint_type)
    
    if str(joint_info[12].decode('utf-8')) == ee_link_name:
        ee_link_id = joint_info[0]
        
    


print(f"active joints: {active_joints}")
print(f"active joint names: {active_joint_names}")
print(f"limits: {limits}")

# Create debug parameters for each active joint
slider_ids = []
for i in range(len(active_joints)):
    joint_name = active_joint_names[i].decode('utf-8')
    joint_type_str = "revolute" if joint_types[i] == 0 else "prismatic"
    param_name = f"{joint_name}: {joint_type_str}"
    lower_limit, upper_limit = limits[i]
    
    # Check if limits are properly defined. If not, use some default
    if lower_limit == upper_limit:
        lower_limit = -3.14
        upper_limit = 3.14
        
    start_val = (lower_limit + upper_limit) / 2.0
    slider_id = p.addUserDebugParameter(param_name, lower_limit, upper_limit, start_val)
    slider_ids.append(slider_id)

print("Use the PyBullet GUI sliders to move the joints. You can also type the exact value next to each slider.")

text_id = -1
line_x_id = -1
line_y_id = -1
line_z_id = -1

step_counter = 0

try:
    while True:
        # Update GUI elements and read sliders at 30Hz instead of 240Hz to prevent lag
        if step_counter % 8 == 0:
            for i in range(len(active_joints)):
                target_pos = p.readUserDebugParameter(slider_ids[i])
                p.setJointMotorControl2(boxId, active_joints[i], p.POSITION_CONTROL, targetPosition=target_pos, force=100)
                
            if ee_link_id is not None:
                # Get end effector state
                link_state = p.getLinkState(boxId, ee_link_id)
                ee_pos = link_state[0]
                ee_quat = link_state[1]
                ee_euler = p.getEulerFromQuaternion(ee_quat)
                
                # Draw frame axes
                axis_len = 0.1
                x_end, _ = p.multiplyTransforms(ee_pos, ee_quat, [axis_len, 0, 0], [0, 0, 0, 1])
                y_end, _ = p.multiplyTransforms(ee_pos, ee_quat, [0, axis_len, 0], [0, 0, 0, 1])
                z_end, _ = p.multiplyTransforms(ee_pos, ee_quat, [0, 0, axis_len], [0, 0, 0, 1])
                
                if line_x_id < 0:
                    line_x_id = p.addUserDebugLine(ee_pos, x_end, lineColorRGB=[1, 0, 0], lineWidth=2)
                    line_y_id = p.addUserDebugLine(ee_pos, y_end, lineColorRGB=[0, 1, 0], lineWidth=2)
                    line_z_id = p.addUserDebugLine(ee_pos, z_end, lineColorRGB=[0, 0, 1], lineWidth=2)
                else:
                    line_x_id = p.addUserDebugLine(ee_pos, x_end, lineColorRGB=[1, 0, 0], lineWidth=2, replaceItemUniqueId=line_x_id)
                    line_y_id = p.addUserDebugLine(ee_pos, y_end, lineColorRGB=[0, 1, 0], lineWidth=2, replaceItemUniqueId=line_y_id)
                    line_z_id = p.addUserDebugLine(ee_pos, z_end, lineColorRGB=[0, 0, 1], lineWidth=2, replaceItemUniqueId=line_z_id)
                
        p.stepSimulation()
        time.sleep(1./240.)
        step_counter += 1
except KeyboardInterrupt:
    pass

p.disconnect()
