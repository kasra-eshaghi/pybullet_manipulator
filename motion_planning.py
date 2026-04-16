# I want to have a function that takes in a start position, end position, and the robot, and plans a path from start to end. The returned path is a sequence of waypoints
import numpy as np
import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt

class MotionPlanner:
    def __init__(self):
        self.traj_qi = None
        self.traj_qf = None
        self.traj_T = None

    def plan_traj_joint_space(self, qi:np.ndarray, qf:np.ndarray, T: float) -> np.ndarray:
        self.traj_qi = qi
        self.traj_qf = qf
        self.traj_T = T

        return
        
    def query_traj(self, t:float) -> np.ndarray:
        self.theta_t = self.traj_qi + (self.traj_qf - self.traj_qi) * (3*np.power(t, 2)/np.power(self.traj_T, 2) - 2*np.power(t, 3)/np.power(self.traj_T, 3))
        self.theta_dot_t = (self.traj_qf - self.traj_qi) * (6*t/np.power(self.traj_T, 2) - 6*np.power(t, 2)/np.power(self.traj_T, 3))
        self.theta_dot_dot_t = (self.traj_qf - self.traj_qi) * (6/np.power(self.traj_T, 2) - 12*t/np.power(self.traj_T, 3))
        
        return self.theta_t, self.theta_dot_t, self.theta_dot_dot_t
        


def main():

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,0]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    boxId = p.loadURDF("franka_panda/panda.urdf",startPos, startOrientation, useFixedBase=True)

    active_joints = []
    active_joint_names = []
    limits = []

    for i in range(p.getNumJoints(boxId)):
        joint_info = p.getJointInfo(boxId, i)
        joint_type = joint_info[2]
        if joint_type == 0 or joint_type == 1:
            active_joints.append(joint_info[0])
            active_joint_names.append(joint_info[1])
            limits.append(joint_info[8:10])

    joint_states = p.getJointStates(boxId, active_joints)
    qi = np.array([state[0] for state in joint_states])
    

    slider_ids = []
    for i in range(len(active_joints)):
        joint_name = active_joint_names[i].decode('utf-8')
        lower_lim, upper_lim = limits[i]
        if lower_lim == upper_lim:
            lower_lim, upper_lim = -3.14, 3.14
        slider_ids.append(p.addUserDebugParameter(f"Goal: {joint_name}", lower_lim, upper_lim, qi[i]))
        
    go_button_id = p.addUserDebugParameter("Go!", 1, 0, 0)
    prev_btn_val = p.readUserDebugParameter(go_button_id)
    
    motion_planner = MotionPlanner()
    
    print("Welcome! Use the PyBullet GUI sliders to set a goal position, then press 'Go!'")
    qs = None
    
    try:
        while True:
            btn_val = p.readUserDebugParameter(go_button_id)
            if btn_val > prev_btn_val:
                prev_btn_val = btn_val
                
                # Get start state from current robot configuration
                joint_states = p.getJointStates(boxId, active_joints)
                qi = np.array([state[0] for state in joint_states])
                
                # Get goal state from sliders
                qf = np.array([p.readUserDebugParameter(sid) for sid in slider_ids])
                
                print("Planning and executing trajectory...")
                traj_time = 2.0
                motion_planner.plan_traj_joint_space(qi, qf, traj_time)
                
                current_time = 0
                qs = np.empty((0,len(qi)))
                while current_time <= traj_time:
                    theta_t, theta_dot_t, theta_dot_dot_t = motion_planner.query_traj(current_time)
                    p.setJointMotorControlArray(boxId, active_joints, p.VELOCITY_CONTROL, targetVelocities=theta_dot_t, targetPositions = theta_t, forces=np.ones(len(active_joints))*100)
                    p.stepSimulation()
                    time.sleep(1./240.)
                    current_time += 1./240.

                    joint_states = p.getJointStates(boxId, active_joints)
                    q_t = np.array([state[0] for state in joint_states])
                    qs = np.vstack((qs, q_t))
                    
                print("Trajectory completed! Choose another goal and press Go.")
            else:
                p.stepSimulation()
                time.sleep(1./240.)
                
    except KeyboardInterrupt:
        pass

    p.disconnect()

    if qs is not None:
        print("Last trajectory joint states:")
        print(qs)
        plt.plot(qs)
        plt.title("Joint Trajectories (Last Run)")
        plt.xlabel("Simulation Timesteps")
        plt.ylabel("Joint Angles (rad)")
        plt.show()
    


if __name__ == "__main__":
    main()

        