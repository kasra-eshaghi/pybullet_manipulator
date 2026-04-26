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
    def __init__(self, urdf_path, ee_link_name, base_pos=[0, 0, 0], base_orn=[0, 0, 0, 1], config_path="rrt_config.yaml"):
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
                
        self.ee_link_id = None
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if str(joint_info[12].decode('utf-8')) == ee_link_name:
                self.ee_link_id = joint_info[0]
        if self.ee_link_id is None:
            raise ValueError(f"End effector link '{ee_link_name}' not found in URDF.")
        
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
        self.ik_distinct_threshold = planner_cfg.get('ik_distinct_threshold', 0.2)
        self.constraint_pos_tol = planner_cfg.get('constraint_pos_tol', 0.02)
        self.constraint_orn_tol = planner_cfg.get('constraint_orn_tol', 0.05)
        
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
        
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -np.array(q2)
            dot = -dot
        if dot > 0.9995:
            res = np.array(q1) + t * (np.array(q2) - np.array(q1))
            return res / np.linalg.norm(res)
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * np.array(q1)) + (s1 * np.array(q2))
        
    def check_constrained_path(self, q1: np.ndarray, q2: np.ndarray, path_func, project_func) -> bool:
        dist = self.distance(q1, q2)
        n_steps = max(2, int(dist / self.collision_res))
        for i in range(1, n_steps + 1):
            t = i / n_steps
            q_interp = q1 + t * (q2 - q1)
            
            # 1. State collision
            if self.check_state_collision(q_interp):
                return True
                
            # 2. Constraint validation (forward kinematics vs theoretical path)
            self.set_robot_state(q_interp)
            fk_state = p.getLinkState(self.robot_id, self.ee_link_id, physicsClientId=self.client_id)
            fk_pos, fk_quat = np.array(fk_state[0]), np.array(fk_state[1])
            
            s_interp = project_func(fk_pos)
            target_pos, target_quat = path_func(s_interp)
            
            pos_err = np.linalg.norm(fk_pos - target_pos)
            if pos_err > self.constraint_pos_tol:
                return True
                
            # Orn error
            dot = np.clip(np.abs(np.dot(fk_quat, target_quat)), 0.0, 1.0)
            orn_err = 2 * np.arccos(dot)
            if orn_err > self.constraint_orn_tol:
                return True
                
        return False

    def sample(self, goal_qs: list) -> np.ndarray:
        if random.random() < self.goal_bias and goal_qs:
            return random.choice(goal_qs)
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

    def plan(self, start_q: np.ndarray, goal_qs: list) -> list:
        start_node = Node(start_q)
        tree = [start_node]
        
        if self.check_state_collision(start_q):
            print("Start state is in collision!")
            return []
            
        if not goal_qs:
            print("No valid target states provided!")
            return []

        print(f"Planning RRT* path across {len(goal_qs)} target IK distributions (max_iter: {self.max_iter})...")
        
        for i in range(self.max_iter):
            # generate new node
            q_rand = self.sample(goal_qs)
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
                if self.enable_early_exit:
                    for idx, gq in enumerate(goal_qs):
                        if self.distance(new_node.q, gq) < self.goal_threshold:
                            print(f"Goal {idx} reached early at iteration {i}!")
                            return self._finalize_path(tree, goal_qs, start_q)

        return self._finalize_path(tree, goal_qs, start_q)

    def _finalize_path(self, tree, goal_qs, start_q):
        # Find the node within goal_threshold that has the lowest cost
        goal_node = None
        min_cost = float('inf')
        best_gq = None
        
        for gq in goal_qs:
            for node in tree:
                dist = self.distance(node.q, gq)
                if dist <= self.goal_threshold:
                    if node.cost < min_cost:
                        goal_node = node
                        min_cost = node.cost
                        best_gq = gq
                        
        if goal_node is None:
            print("Failed to find path to goal!")
            return []
        
        # Try to connect goal_node exactly to best_gq directly if possible
        if self.distance(goal_node.q, best_gq) > 1e-6:
            if not self.check_path_collision(goal_node.q, best_gq):
                final_node = Node(best_gq)
                final_node.parent = goal_node
                final_node.cost = goal_node.cost + self.distance(goal_node.q, best_gq)
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

    def _get_ik_solutions(self, target_pos, target_quat, retries: int):
        ll_list = self.lower_limits.tolist()
        ul_list = self.upper_limits.tolist()
        jr_list = (self.upper_limits - self.lower_limits).tolist()
        movable_joints = self.active_joints
        
        valid_goal_qs = []
        for retry in range(retries):
            rp_list = []
            for ll, ul in zip(ll_list, ul_list):
                if retry == 0:
                    rp_list.append((ll + ul) / 2.0)
                else:
                    rp_list.append(random.uniform(ll, ul))
                    
            for temp_i, temp_q in zip(movable_joints, rp_list):
                 p.resetJointState(self.robot_id, temp_i, temp_q, physicsClientId=self.client_id)

            ik_sol = p.calculateInverseKinematics(
                self.robot_id, self.ee_link_id, target_pos, target_quat,
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
                    
            if self.check_state_collision(qf):
                 continue
                 
            is_distinct = True
            for existing_q in valid_goal_qs:
                if self.distance(qf, existing_q) < self.ik_distinct_threshold:
                    is_distinct = False
                    break
                    
            if is_distinct:
                valid_goal_qs.append(qf)
                
        return valid_goal_qs

    def plan_t_space(self, start_q: np.ndarray, t_pos: list, t_euler: list) -> list:
        target_quat = p.getQuaternionFromEuler(t_euler)
        valid_goal_qs = self._get_ik_solutions(t_pos, target_quat, self.ik_retries)
                
        if not valid_goal_qs:
            print("Failed to pull any valid collision-free IK distributions from Null-Space.")
            return []
        else:
            print(f"Valid Distinct IK Solution Extracted: {len(valid_goal_qs)} solutions")
            
        return self.plan(start_q, valid_goal_qs)

    def plan_constrained(self, start_qs: list, goal_qs: list, path_func, project_func) -> dict:
        tree = []
        for sq in start_qs:
            tree.append(Node(sq))
            
        print(f"Planning Constrained Task Space RRT* ({len(start_qs)} start configs -> {len(goal_qs)} goal configs, max_iter: {self.max_iter})...")
        
        for i in range(self.max_iter):
            s_rand = random.uniform(0.0, 1.0)
            target_pos, target_quat = path_func(s_rand)
            
            try_goal = False
            if random.random() < self.goal_bias and goal_qs:
                q_rand = random.choice(goal_qs)
                try_goal = True
            else:
                q_rands = self._get_ik_solutions(target_pos, target_quat, retries=1)
                if not q_rands:
                    continue
                q_rand = random.choice(q_rands)
                
            nearest_node = min(tree, key=lambda node: self.distance(node.q, q_rand))
            
            dist = self.distance(nearest_node.q, q_rand)
            if dist < self.step_size:
                q_new = q_rand
            else:
                scale = self.step_size / dist
                q_new = nearest_node.q + scale * (q_rand - nearest_node.q)
                
            if not self.check_constrained_path(nearest_node.q, q_new, path_func, project_func):
                new_node = Node(q_new)
                
                neighbors = self.get_neighbors(tree, new_node)
                
                min_cost_node = nearest_node
                min_cost = nearest_node.cost + self.distance(nearest_node.q, new_node.q)
                for neighbor in neighbors:
                    cost = neighbor.cost + self.distance(neighbor.q, new_node.q)
                    if cost < min_cost:
                        if not self.check_constrained_path(neighbor.q, new_node.q, path_func, project_func):
                            min_cost_node = neighbor
                            min_cost = cost
                            
                new_node.parent = min_cost_node
                new_node.cost = min_cost
                tree.append(new_node)
                
                for neighbor in neighbors:
                    rewired_cost = new_node.cost + self.distance(new_node.q, neighbor.q)
                    if rewired_cost < neighbor.cost:
                        if not self.check_constrained_path(new_node.q, neighbor.q, path_func, project_func):
                            neighbor.parent = new_node
                            neighbor.cost = rewired_cost
                            
        # Post-process tree to extract all valid root-mapped paths
        valid_paths = {}
        for node in tree:
            for gq in goal_qs:
                if self.distance(node.q, gq) <= self.goal_threshold:
                    if not self.check_constrained_path(node.q, gq, path_func, project_func):
                        # Construct valid path backwards
                        path = [gq]
                        curr = node
                        while curr is not None:
                            path.append(curr.q)
                            curr = curr.parent
                        path.reverse()
                        
                        root_tuple = tuple(path[0])
                        c_cost = 0.0
                        for c_i in range(len(path)-1):
                            c_cost += self.distance(path[c_i], path[c_i+1])
                        
                        # Cache the most optimal route identified originating from this specific root
                        if root_tuple not in valid_paths:
                            valid_paths[root_tuple] = (path, c_cost)
                        else:
                            if c_cost < valid_paths[root_tuple][1]:
                                valid_paths[root_tuple] = (path, c_cost)
                                
        final_dict = {}
        for r_key, (path, c_cost) in valid_paths.items():
            final_dict[r_key] = self.smooth_constrained_path(path, path_func, project_func)
            
        print(f"Constrained exploration concluded. Discovered spanning routes tied to {len(final_dict)} independent start configurations!")
        return final_dict

    def plan_constrained_t_space(self, start_pos, start_euler, goal_pos, goal_euler) -> dict:
        start_pos = np.array(start_pos)
        start_quat = p.getQuaternionFromEuler(start_euler)
        goal_pos = np.array(goal_pos)
        goal_quat = p.getQuaternionFromEuler(goal_euler)
        
        def path_func(s):
            s = np.clip(s, 0.0, 1.0)
            p_interp = start_pos + s * (goal_pos - start_pos)
            q_interp = self._slerp(start_quat, goal_quat, s)
            return p_interp, q_interp
            
        def project_func(pos):
            vec_a = start_pos
            vec_b = goal_pos
            vec_p = np.array(pos)
            
            line_vec = vec_b - vec_a
            sq_len = np.dot(line_vec, line_vec)
            if sq_len < 1e-8:
                return 0.0
                
            s = np.dot(vec_p - vec_a, line_vec) / sq_len
            return float(np.clip(s, 0.0, 1.0))
            
        print("Pre-validating constrained task-space feasibility...")
        for s in np.linspace(0, 1, 25):
            p_s, q_s = path_func(s)
            sol = self._get_ik_solutions(p_s, q_s, retries=max(1, self.ik_retries))
            if not sol:
                print(f"Task space path is fully occluded or mathematically unreachable at s={s:.2f}. Aborting.")
                return {}
                
        print("Feasibility check passed. Acquiring start and goal multi-roots...")
        start_qs = self._get_ik_solutions(start_pos, start_quat, retries=self.ik_retries*2)
        goal_qs = self._get_ik_solutions(goal_pos, goal_quat, retries=self.ik_retries*2)
        
        if not start_qs or not goal_qs:
            print("Failed to secure valid multi-root clusters for either start or goal. Aborting.")
            return {}
            
        return self.plan_constrained(start_qs, goal_qs, path_func, project_func)

    def get_tree_cartesian_nodes(self):
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
                pos = p.getLinkState(self.robot_id, self.ee_link_id, physicsClientId=self.client_id)[0]
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

    def smooth_constrained_path(self, path: list, path_func, project_func) -> list:
        if len(path) <= 2:
            return path
            
        print("Smoothing constrained path...")
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
            
            if not self.check_constrained_path(q_i, q_j, path_func, project_func):
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
                
        print(f"Smoothed constrained path down to {len(smoothed_path)} waypoints")
        return smoothed_path

    def generate_trajectory(self, path: list, total_time: float, use_smoothing: bool = False):
        if not path:
            return None
            
        num_waypoints = len(path)
        if num_waypoints == 1:
             return lambda t: (path[0], np.zeros_like(path[0]), np.zeros_like(path[0]), path[0], np.zeros_like(path[0]), np.zeros_like(path[0]), 0.0)
             
        distances = []
        for i in range(num_waypoints - 1):
            distances.append(self.distance(path[i], path[i+1]))
            
        total_dist = sum(distances)
        if total_dist == 0:
             return lambda t: (path[0], np.zeros_like(path[0]), np.zeros_like(path[0]), path[0], np.zeros_like(path[0]), np.zeros_like(path[0]), 0.0)
             
        times = [0.0]
        for dist in distances:
            times.append(times[-1] + total_time * (dist / total_dist))
            
        def evaluate(t: float):
            t = np.clip(t, 0.0, total_time)
            
            if use_smoothing:
                # Option 1: Map overall timeline mapping across a seamless 3rd-order spatial profile globally
                t_norm = t / total_time
                t_virtual_norm = 3 * (t_norm**2) - 2 * (t_norm**3)
                t_virtual = t_virtual_norm * total_time
                
                # Native derivatives over Global Space (Chain Rules)
                dt_virt_dt = (6 * t_norm - 6 * (t_norm**2)) / total_time * total_time
                d2t_virt_dt2 = (6 - 12 * t_norm) / (total_time**2) * total_time
            else:
                # Execution halts implicitly at boundaries (isolated segmented interpolation)
                t_virtual = t
                dt_virt_dt = 1.0
                d2t_virt_dt2 = 0.0
            
            idx = 0
            for i in range(num_waypoints - 1):
                if t_virtual <= times[i+1]:
                    idx = i
                    break
            
            if t_virtual >= times[-1]:
                idx = num_waypoints - 2
                
            qi = path[idx]
            qf = path[idx+1]
            ti = times[idx]
            tf = times[idx+1]
            T = tf - ti
            
            if T <= 1e-6:
                return qf, np.zeros_like(qf), np.zeros_like(qf), qf, np.zeros_like(qf), np.zeros_like(qf), float(idx + 1)
                
            tau = t_virtual - ti
            tau_n = tau / T
            
            # Geometric Path Derivations (vs parameter s where segment is strictly idx -> idx+1)
            qs_dot = qf - qi  # dq/ds 
            qs_ddot = np.zeros_like(qf) # d2q/ds2
            
            if not use_smoothing:
                # 3rd Order Polynomial timing mapping bounds to s-space LOCALLY
                tau_poly = 3 * (tau_n**2) - 2 * (tau_n**3)
                s = float(idx + tau_poly)
                
                dtau_poly_dt = (6 * tau_n - 6 * (tau_n**2)) / T
                d2tau_poly_dt2 = (6 - 12 * tau_n) / (T**2)
                
                qt = qi + qs_dot * tau_poly
                qs = qt
                
                qt_dot = qs_dot * dtau_poly_dt
                qt_ddot = qs_dot * d2tau_poly_dt2
            else:
                # Global mapping execution (Linear topological traversing driven by global mapped timing!)
                s = float(idx + tau_n)
                
                dq_dt_virt = qs_dot / T
                
                qt = qi + qs_dot * tau_n
                qs = qt
                
                qt_dot = dq_dt_virt * dt_virt_dt
                qt_ddot = 0.0 + dq_dt_virt * d2t_virt_dt2
            
            return qt, qt_dot, qt_ddot, qs, qs_dot, qs_ddot, s
            
        return evaluate
