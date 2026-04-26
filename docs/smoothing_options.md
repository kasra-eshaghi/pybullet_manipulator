# Continuous Motion Smoothing Options for Linearly Segmented Trajectories

When an End-Effector moves across dynamically generated Cartesian path segments via Inverse Kinematics, dealing with structural joints abruptly colliding with mathematical waypoints is a major robotics control problem. This file documents the primary options for achieving smooth motion along topological configurations.

## 1. Global Non-Stop Path Timing (Selected)
Instead of forcing a mathematical velocity curve (like a 3rd order polynomial) into every distinct path segment (which forces the motors to start at 0 speed and end at 0 speed repeatedly), a single continuous 3rd-order curve is mapped directly onto a virtual timeline spanning the entire execution sequence. 

* **The Math**: The virtual time accelerates gracefully over the entire duration seamlessly dragging the execution pointer linearly through the segments. The robot does not decelerate between segments because $dt_{virtual}/dt$ is uninterrupted over local node horizons.
* **The Drawback**: When a physical system travels rapidly through mathematical corners, its joints change directions identically rapidly. In simulation space, physics engines can safely interpret "infinite step change" jerks in acceleration dynamically. In physical hardware, this creates abrupt violent snapping!

## 2. Geometric Spline Interpolation (B-Splines/Bezier)
Converting the raw RRT nodes into an optimized curving path geometry across the joint states directly guarantees smooth derivative continuity (1st and 2nd Order hardware safety bounds) organically.
* **The Math**: Control nodes shape a unified mathematical geometric curve mathematically guaranteeing smooth transitions throughout the local topology matrix!
* **The Drawback**: Curving $N$-dimensional joint spaces organically causes nonlinear bulging out inside the absolute Cartesian reference frame (Task Space). Meaning: although the joints glide beautifully, the End Effector will physically stray, drifting heavily off the restricted target paths and violating $0.005$m tolerances. 

## 3. Parabolic Trajectory Blends (Trapezoidal Profiles)
Standard integration for modern commercial hardware: preserving strictly straight local trajectories but dynamically overwriting transition nodes with mathematically constrained smoothing loops.
* **The Math**: Constant fast kinematic velocity bounds are held safely. As the manipulator targets an upcoming boundary, the profile initiates early cornering across local $ds/dt$ vectors blending the physical momentum continuously to escape zero-velocity states gracefully!
* **The Drawback**: Because mathematical blends intrinsically round corners, path tolerances degrade gracefully proportional to blend speeds natively. Overly smoothed systems might technically curve straight out of your constraint fields!

## 4. True Cartesian Control via Jacobian Inversion
The ultimate deterministic industrial solution. 
* **The Math**: Throw away joint space parameters completely. Define the specific topological vectors statically inside physical reality frame matrices, evaluating target vectors exactly over $x(t)$. Continually drive control arrays functionally natively using exactly $q_{dot} = J^+ \cdot x_{dot}$.
* **The Drawback**: Entirely rewrites native RRT mapping implementations. Requires transitioning strictly into robust kinematics-velocity controllers versus directly applying Inverse Kinematics matrices targeting strict topological coordinates across sequential nodes.
