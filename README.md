# Interactive Constrained Robot Motion Planning Simulator

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyBullet](https://img.shields.io/badge/PyBullet-Physics_Engine-blue?style=for-the-badge)
![Robotics](https://img.shields.io/badge/Robotics-Path_Planning-brightgreen?style=for-the-badge)

A sophisticated trajectory planning simulator aimed at solving high-tolerance industrial motion tasks (such as robotic welding). Built natively on top of the **PyBullet** physics engine, this project implements a highly modified, projection-based **Task-Space Constrained RRT*** algorithm to tightly orchestrate a highly non-linear, 6-DOF serial manipulator. 

## 🚀 Key Technical Features

### 1. Dual-Stage Backward-Propagated Planning Pipeline
Typical planners struggle to merge free-form transit routing with strict geometric tasks. This simulator relies on a custom architectural inversion: 
* **Stage 2 (Constrained Task)** is solved *first*, mapping comprehensive multi-root Inverse Kinematic (IK) domains to ensure valid execution entry.
* **Stage 1 (Unconstrained Transit)** uses dynamic target clustering to route standard joint-space RRT* pathways sequentially matching the discovered roots of Stage 2.

### 2. Projection-Based Mathematical Drift Mitigation
Classical constraint algorithms often linearly interpolate constraints over distances, resulting in mathematical "drifting" (due to joint-space Jacobian non-linearities). This architecture utilizes an analytic Orthogonal Vector Projector, permanently preventing geometric path sliding and enforcing an absolute tolerance bound on the End Effector. 

### 3. Trajectory Physics Discretization
Swapped out baseline polynomial velocity tracking (Cubic Splines) for raw continuous **Trapezoidal Linear Interpolators**. Since robots executing strict geometric operations (like welding) require flat continuous velocities rather than smoothed braking, the engine extracts exact physical hardware arrays without parabolic acceleration overlaps.

### 4. Interactive Simulation & Analytics Head
Built with native PyBullet Debug tools—users can interactively drag topological Start and Goal target boundary widgets across the simulation grid. Upon command, the engine dynamically instantiates its search trees, generates PyBullet debug line traces, and mathematically executes real physics evaluations against the hardware limits directly.

## 🛠️ Built With
* **Python 3**
* **PyBullet** (Physics constraint solving and visualization backend)
* **NumPy** (Kinematic matrix projection and topological math)
* **Matplotlib** *(Optional)* (Used for deep analytical tracing of temporal and spatial Joint Velocities via numerical differentiation)

## ⚙️ Quick Start

Install required engine dependencies:
```bash
pip install pybullet numpy
```

Launch the interactive RRT* Constrained execution environment:
```bash
python interactive_constrained_rrt.py
```
> **Usage Note**: Upon launching, adjust the Start (Green) and Goal (Red) target nodes using PyBullet's native slider/drag interfaces. Click the `Plan and Go` execution parameter to dispatch the robotic solver! 
