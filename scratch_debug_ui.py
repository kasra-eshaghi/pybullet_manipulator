import pybullet as p
import time

p.connect(p.GUI)
b1 = p.addUserDebugParameter("Button 1", 1, 0, 0)
time.sleep(1)
p.removeUserDebugItem(b1)
while True:
    p.stepSimulation()
    time.sleep(1./240.)
