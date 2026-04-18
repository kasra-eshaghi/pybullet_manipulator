import numpy as np
import pybullet as p
import pybullet_data
import random
import yaml
import time

class Node:
    def __init__(self, q: np.ndarray):
        self.q = q
        self.parent = None
        self.cost = 0.0

class RRTPlanner:
    def __init__(self, urdf_path, base_pos=[0, 0, 0], base_orn=[0, 0, 0, 1], config_path="rrt_config.yaml"):
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        
        self.robot_id = p.loadURDF(urdf_path, basePosition=base_pos, baseOrientation=base_orn, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.client_id)
        

        # find active joints
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        self.active_joints = []
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if info[2] == p.JOINT_REVOLUTE or info[2] == p.JOINT_PRISMATIC:
                self.active_joints.append(i)
        
        self.obstacle_ids = []
        self.dynamic_obs_id = None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        planner_cfg = config.get('planner', {})
        self.max_iter = planner_cfg.get('max_iterations', 2000)
        self.step_size = planner_cfg.get('step_size', 0.1)
        self.goal_bias = planner_cfg.get('goal_bias', 0.05)
        self.rewire_radius = planner_cfg.get('rewire_radius', 0.5)
        self.collision_res = planner_cfg.get('collision_check_resolution', 0.05)
        self.smooth_iter = planner_cfg.get('smoothing_iterations', 150)
        self.goal_threshold = planner_cfg.get('goal_threshold', 0.1)
        self.enable_early_exit = planner_cfg.get('early_exit', True)
        self.ik_retries = planner_cfg.get('ik_retries', 5)
        
        weights = config.get('joint_weights', [])
        # pad weights if needed
        self.weights = np.ones(len(self.active_joints))
        for i in range(min(len(weights), len(self.active_joints))):
            self.weights[i] = weights[i]
            
        self.lower_limits = np.zeros(len(self.active_joints))
        self.upper_limits = np.zeros(len(self.active_joints))
        
        custom_limits = planner_cfg.get('custom_limits', [])
        
        for i, joint_idx in enumerate(self.active_joints):
            info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client_id)
            ll, ul = info[8], info[9]
            if ul <= ll:
                ll, ul = -3.14159, 3.14159
                
            if i < len(custom_limits) and custom_limits[i] is not None:
                ll, ul = custom_limits[i]
                
            self.lower_limits[i] = ll
            self.upper_limits[i] = ul

        self._build_adjacency_matrix()

    def update_dynamic_obstacle(self, pos, half_extents):
        """Updates or spawns a dynamic box obstacle in the DIRECT collision engine"""
        if self.dynamic_obs_id is not None:
            p.removeBody(self.dynamic_obs_id, physicsClientId=self.client_id)
        
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
        self.dynamic_obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, basePosition=pos, physicsClientId=self.client_id)

    def _build_adjacency_matrix(self):
        num_links = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        self.link_parents = {}
        for i in range(num_links):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            parent = info[16]
            self.link_parents[i] = parent
            
    def _are_links_close_kinematically(self, linkA, linkB, max_dist=2):
        pathA = [linkA]
        curr = linkA
        while curr in self.link_parents and self.link_parents[curr] != -1:
            curr = self.link_parents[curr]
            pathA.append(curr)
        pathA.append(-1)
        
        pathB = [linkB]
        curr = linkB
        while curr in self.link_parents and self.link_parents[curr] != -1:
            curr = self.link_parents[curr]
            pathB.append(curr)
        pathB.append(-1)
        
        for i, nodeA in enumerate(pathA):
            if nodeA in pathB:
                j = pathB.index(nodeA)
                dist = i + j
                if dist <= max_dist:
                    return True
        return False

    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        return float(np.linalg.norm(self.weights * (q1 - q2)))
        
    def set_robot_state(self, q: np.ndarray):
        for i, joint_idx in enumerate(self.active_joints):
            p.resetJointState(self.robot_id, joint_idx, q[i], physicsClientId=self.client_id)
            
    def check_state_collision(self, q: np.ndarray) -> bool:
        # 1. Check joint limits
        for i in range(len(self.active_joints)):
            if q[i] < self.lower_limits[i] or q[i] > self.upper_limits[i]:
                return True
                
        # 2. Check collision
        self.set_robot_state(q)
        p.performCollisionDetection(physicsClientId=self.client_id)
        
        pts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id, physicsClientId=self.client_id)
        for pt in pts:
            linkA = pt[3]
            linkB = pt[4]
            if not self._are_links_close_kinematically(linkA, linkB, max_dist=2):
                return True
            
        if self.dynamic_obs_id is not None:
             pts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.dynamic_obs_id, physicsClientId=self.client_id)
             if len(pts) > 0:
                 return True
                
        return False
        
    def check_path_collision(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        dist = self.distance(q1, q2)
        n_steps = max(2, int(dist / self.collision_res))
        for i in range(1, n_steps + 1):
            t = i / n_steps
            q_interp = q1 + t * (q2 - q1)
            if self.check_state_collision(q_interp):
                return True
        return False

    def sample(self, goal_q: np.ndarray) -> np.ndarray:
        if random.random() < self.goal_bias:
            return goal_q
        else:
            return np.random.uniform(self.lower_limits, self.upper_limits)
            
    def nearest(self, tree: list, q: np.ndarray) -> Node:
        return min(tree, key=lambda node: self.distance(node.q, q))
        
    def steer(self, from_node: Node, to_q: np.ndarray) -> np.ndarray:
        dist = self.distance(from_node.q, to_q)
        if dist < self.step_size:
            return to_q
        else:
            direction = to_q - from_node.q
            scale = self.step_size / dist
            return from_node.q + scale * direction

    def get_neighbors(self, tree: list, new_node: Node) -> list:
        neighbors = []
        for node in tree:
            if self.distance(node.q, new_node.q) < self.rewire_radius:
                neighbors.append(node)
        return neighbors

    def plan(self, start_q: np.ndarray, goal_q: np.ndarray) -> list:
        start_node = Node(start_q)
        tree = [start_node]
        
        if self.check_state_collision(start_q):
            print("Start state is in collision!")
            return []
        if self.check_state_collision(goal_q):
            print("Goal state is in collision!")
            return []

        print(f"Planning RRT* path (max_iter: {self.max_iter})...")
        
        for i in range(self.max_iter):
            # generate new node
            q_rand = self.sample(goal_q)
            nearest_node = self.nearest(tree, q_rand)
            q_new = self.steer(nearest_node, q_rand)
            
            # check if new node is valid
            # TODO: Change check_path_collision to make sure all points along the path (start and end) are: (1) collision free and (2) within joint limits (currently only checks for collisions)
            if not self.check_path_collision(nearest_node.q, q_new):
                new_node = Node(q_new)

                # find feasible neighbors within range
                neighbors = self.get_neighbors(tree, new_node)
                
                # find best neighbor
                min_cost_node = nearest_node
                min_cost = nearest_node.cost + self.distance(nearest_node.q, new_node.q)
                for neighbor in neighbors:
                    cost = neighbor.cost + self.distance(neighbor.q, new_node.q)
                    if cost < min_cost:
                        if not self.check_path_collision(neighbor.q, new_node.q):
                            min_cost_node = neighbor
                            min_cost = cost
                            
                new_node.parent = min_cost_node
                new_node.cost = min_cost
                tree.append(new_node)
                
                # rewire neighbors
                for neighbor in neighbors:
                    rewired_cost = new_node.cost + self.distance(new_node.q, neighbor.q)
                    if rewired_cost < neighbor.cost:
                        if not self.check_path_collision(new_node.q, neighbor.q):
                            neighbor.parent = new_node
                            neighbor.cost = rewired_cost
                            
                # Check early exit criteria
                if self.enable_early_exit and self.distance(new_node.q, goal_q) < self.goal_threshold:
                    print(f"Goal reached early at iteration {i}!")
                    break


        # Find the node within goal_threshold that has the lowest cost
        goal_node = None
        min_cost = float('inf')
        for node in tree:
            dist = self.distance(node.q, goal_q)
            if dist <= self.goal_threshold:
                if node.cost < min_cost:
                    goal_node = node
                    min_cost = node.cost
                        
        if goal_node is None:
            print("Failed to find path to goal!")
            return []
        
        # Try to connect goal_node exactly to goal_q directly if possible
        if self.distance(goal_node.q, goal_q) > 1e-6:
            if not self.check_path_collision(goal_node.q, goal_q):
                final_node = Node(goal_q)
                final_node.parent = goal_node
                final_node.cost = goal_node.cost + self.distance(goal_node.q, goal_q)
                tree.append(final_node)
                goal_node = final_node
        
        final_node = goal_node
        
        # path planning
        path = []
        curr = final_node
        while curr is not None:
            path.append(curr.q)
            curr = curr.parent
            
        path.reverse()
        print(f"Initial path found with {len(path)} waypoints")
        self.set_robot_state(start_q)
        self.last_tree = tree

        # smooth path
        smooth_path = self.smooth_path(path)

        return smooth_path

    def plan_t_space(self, start_q: np.ndarray, ee_link_id: int, t_pos: list, t_euler: list) -> list:
        # Fetch IK limits constraints directly from our active joints limits
        ll_list = []
        ul_list = []
        jr_list = []
        movable_joints = []
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if info[2] != p.JOINT_FIXED:
                movable_joints.append(i)
                ll, ul = info[8], info[9]
                if ll >= ul:
                    ll, ul = -3.14159, 3.14159
                ll_list.append(ll)
                ul_list.append(ul)
                jr_list.append(ul - ll)

        for retry in range(self.ik_retries):
            rp_list = []
            for ll, ul in zip(ll_list, ul_list):
                if retry == 0:
                    rp_list.append((ll + ul) / 2.0)
                else:
                    # Randomize the null-space solver rest pose
                    rp_list.append(random.uniform(ll, ul))
                    
            # Temporarily inject this randomized pose physically so the IK solver roots dynamically
            for temp_i, temp_q in zip(movable_joints, rp_list):
                 p.resetJointState(self.robot_id, temp_i, temp_q, physicsClientId=self.client_id)

            ik_sol = p.calculateInverseKinematics(
                self.robot_id, ee_link_id, t_pos, p.getQuaternionFromEuler(t_euler),
                lowerLimits=ll_list,
                upperLimits=ul_list,
                jointRanges=jr_list,
                restPoses=rp_list,
                maxNumIterations=100,
                residualThreshold=1e-5,
                physicsClientId=self.client_id
            )

            qf = np.zeros(len(self.active_joints))
            for i, joint_idx in enumerate(self.active_joints):
                if joint_idx in movable_joints:
                    ik_idx = movable_joints.index(joint_idx)
                    qf[i] = ik_sol[ik_idx]
                    
            print(f"IK Target Derived (Attempt {retry+1}/{self.ik_retries}): {qf}")
            
            # Perform preemptive hardware collision check to skip 10000 RRT loops if inherently broken
            if self.check_state_collision(qf):
                 print(f"IK Attempt {retry+1} resulted in joint occlusion/collision. Rerolling alternate posture...")
                 continue
                 
            path = self.plan(start_q, qf)
            if path:
                return path
                
            print(f"RRT* failed to connect IK attempt {retry+1}. Rooting new alternative elbow pose...")
            
        print("Exhausted all IK alternative permutations. Path planning comprehensively failed.")
        return []

    def get_tree_cartesian_nodes(self, ee_link_id):
        """Extracts the end-effector Cartesian [X, Y, Z] for every node and its parent in the last planned tree."""
        if not hasattr(self, 'last_tree') or not self.last_tree:
            return [], []
            
        points = []
        lines = [] # (start, end)
        
        # We need to evaluate FK for every unique joint configuration
        q_to_pos = {}
        for node in self.last_tree:
            q_tuple = tuple(node.q)
            if q_tuple not in q_to_pos:
                self.set_robot_state(node.q)
                pos = p.getLinkState(self.robot_id, ee_link_id, physicsClientId=self.client_id)[0]
                q_to_pos[q_tuple] = pos
            points.append(q_to_pos[q_tuple])
            
            if node.parent is not None:
                parent_pos = q_to_pos[tuple(node.parent.q)]
                lines.append((parent_pos, q_to_pos[q_tuple]))
                
        return points, lines

    def smooth_path(self, path: list) -> list:
        if len(path) <= 2:
            return path
            
        print("Smoothing path...")
        smoothed_path = list(path)
        
        for _ in range(self.smooth_iter):
            if len(smoothed_path) <= 2:
                break
            i = random.randint(0, len(smoothed_path) - 2)
            j = random.randint(i + 1, len(smoothed_path) - 1)
            
            if j - i <= 1:
                continue
                
            q_i = smoothed_path[i]
            q_j = smoothed_path[j]
            
            if not self.check_path_collision(q_i, q_j):
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
                
        print(f"Smoothed path down to {len(smoothed_path)} waypoints")
        return smoothed_path

    def generate_trajectory(self, path: list, total_time: float):
        if not path:
            return None
            
        num_waypoints = len(path)
        if num_waypoints == 1:
            return lambda t: (path[0], np.zeros_like(path[0]), np.zeros_like(path[0]))
            
        distances = []
        for i in range(num_waypoints - 1):
            distances.append(self.distance(path[i], path[i+1]))
            
        total_dist = sum(distances)
        if total_dist == 0:
             return lambda t: (path[0], np.zeros_like(path[0]), np.zeros_like(path[0]))
             
        times = [0.0]
        for dist in distances:
            times.append(times[-1] + total_time * (dist / total_dist))
            
        def evaluate(t: float):
            t = np.clip(t, 0.0, total_time)
            
            idx = 0
            for i in range(num_waypoints - 1):
                if t <= times[i+1]:
                    idx = i
                    break
            
            if t >= times[-1]:
                idx = num_waypoints - 2
                
            qi = path[idx]
            qf = path[idx+1]
            ti = times[idx]
            tf = times[idx+1]
            T = tf - ti
            
            if T <= 1e-6:
                return qf, np.zeros_like(qf), np.zeros_like(qf)
                
            tau = t - ti
            
            q_tau = qi + (qf - qi) * (3*np.power(tau, 2)/np.power(T, 2) - 2*np.power(tau, 3)/np.power(T, 3))
            q_dot = (qf - qi) * (6*tau/np.power(T, 2) - 6*np.power(tau, 2)/np.power(T, 3))
            q_ddot = (qf - qi) * (6/np.power(T, 2) - 12*tau/np.power(T, 3))
            
            return q_tau, q_dot, q_ddot
            
        return evaluate
