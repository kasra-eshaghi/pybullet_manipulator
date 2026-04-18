# Purpose
The purpose of this document is to outline the requirements and steps to implementing RRT* motion planning for a serial manipulator in pybullet.

The developed planner should be scalable to any robot, not just the panda. Note that the URDF for a robot may include more joints/links that the minimum.

# Key functions
## Initializer
- Inputs: 
    - robot_body_id (int) from p.loadURDF -> representing the robot
    - obstacle_ids (list of ints) > representing the obstacles
    - All other relevant RRT* planning parameters that you tell me of.   
- Purpose:
    - Initialize the RRT* planner
- Outputs:
    - None

## plan path c space
- Inputs: 
    - goal_joint_config (np.ndarray) -> the goal joint configuration
- Purpose:
    - Plan a path from the robot's current configuration to the goal configuration in the joint space using the RRt* algorithm
- Outputs:
    - path (list of np.ndarrays) -> the path from start to goal
- Assumption:
    - Assumes start point of path is robot's current configuration
    
# Key RRT* components
## Distance checker
- Calculate distance between two joint configurations as the euclidean distance in the joint space.

## Configuration Collision checker
- Use internal pybullet libraries to check for collisions

## Path collision checker
- Sample points along the path and check for collisions using internal pybullet libraries

## Local path planning
- Plan local path between nodes in the tree based on a straight line.

## Post planning smoother
- Smooth the path found by RRT* using iterative shortcutting.

## Trajectory generation
- Once a path is planned, turn it into a trajectory. For this, I will give you the desired robot speed (m/s for the end effector), you will turn this into an approximate travel time, and then use a cubic polynomial to interpolate between the waypoints in the path.
- The trajectory generator will be queried for the robot's joint configuration at a given time t, which wil then be sent to the robot controller (see motion_planning.py for current implementation)

# Steps
1. Identify all RRT* planning parameters that will be needed to initialize the planner, and create a rrt_config.yaml file that I can use to adjust these parameters.
2. Implement the RRT* path planner for the panda robot with self collisions enabled but no other obstacles.
3. Create an interactive pybullet gui where I can set the goal position in joint space, and it plans and executes the path.

motion_planning.py and interactive.py are your reference for how I will be using the robot and pybullet.
