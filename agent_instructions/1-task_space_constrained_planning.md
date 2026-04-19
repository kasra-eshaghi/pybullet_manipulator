# Purpose
The purpose of this document is to outline the requirements for the task space constrained RRT* implementation that you are going to help me build.

Expected deliverables:
1. Updated RRTPlanner class
2. new interactice_constrained_rrt.py file.

## Terminology

- **Task Space**: The space of all possible poses (position and orientation) of the robot's end-effector.
- **Pose**: A specific position and orientation of the robot's end-effector (i.e., in task space).
- **Task Path**: A sequence of poses for the end effector of the robot.

- **Joint Space**: The space of all possible joint configurations of the robot.
- **Configuration**: A specific joint configuration of the robot (i.e., in joint space).
- **Joint Path**: A sequence of configurations for the robot's joints.

## Current Implementation and desired change
The current RRT implementation is focused on moving from a initial configuration to a final pose. 
The next iteration of this planner will focus on planning a path from a starting pose to a goal pose, while constrained to following a specific task path. For example, going from a pose P1 to a pose P2 while constrained to following a straight line path in task space that connects them. 

# Implementation details
## Updated RRTPlanner class
The new planner will take as inputs a starting pose, a goal pose. It will assume, for now, that the path connecting the two is a straight line path (in task space). However, generate the path in a way such that in the future, I can give it a custom path as well. Note that the path is straight in position and orientation. For example, if the orientation between the start and goal poses changesby 45 degrees, then the orientation of the end effector should change linearly from the start to the goal orientation. 

The planner will keep the existing RRT* based approach. However, to solve the constrained problem, it will require the following changes:
1. for a given start pose and goal pose, it has to generate a number of random IK solutions for both poses (q_start_list and q_goal_list). These will all be part of the RRT tree, and a solution from any configuuration in q_start_list to any configuration in q_goal_list will be deemed feasible.
2. When sampling, samples should be taken from the task space, and then converted to joint space using IK. The samples should be taken along the straight line path connecting the start and goal poses in task space, NOT in random locations in the task space. Due to the redundancy of IK solutions, each task space sample can be converted to muliple joint configurations for the RRT tree.
3. When checking connectivity between points in the RRT tree (ie through RRTPlanner.check_state_collision), you must also check to make sure the state is on the path. 
4. For any input start and goal pose, the planner should make sure the points along the straight line path connecting them are collision free. If not, it should not plan a path. 
5. The developed method must continue to check for collisions, and should work once obstacles are added to the environment. 

## New interactive script
A new interactive script should be created that allows the user to select a start pose and a goal pose, and then plan a path from the start pose to the goal pose while constrained to following a straight line path in task space. 
The gui should use sliders to allow the user to set the start and goal poses. Furthermore, it should draw the straight line path between the start and goal poses interactively, and only begin planning and execution when the operator presses "go". Take inspiration from the gui in interactive_rrt.py. 

## Testing plan
1. Develop the gui that allows the user to set the start and goal poses, and then plan a path from the start pose to the goal pose while constrained to following a straight line path in task space. 
2. Test the planner with a simple environment with no obstacles. 
3. Test the planner with an environment with obstacles. 

# Coding objectives
- Try to implement the new code within the existing structure, so that functions are generalizable. However, if it gets too messy, then make custom functions.