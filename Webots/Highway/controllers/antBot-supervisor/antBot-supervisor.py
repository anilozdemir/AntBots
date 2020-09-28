# This controller is for robot to move.
# The supervisor can control this robot.

from controller import Robot, Motor, Camera, Supervisor
import numpy as np

# INIT #
TIMESTEP = 32
supervisor  = Supervisor()
robot       = supervisor.getFromDef("antBot")
trans_field = robot.getField("translation")
values      = trans_field.getSFVec3f()
print("Pos: {:.2f} {:.2f} {:.2f}".format(*values))

# Main loop:
INITIAL = [-7, 0.5, 5]
trans_field.setSFVec3f(INITIAL)

print('>> Start')
t = supervisor.getTime()
while supervisor.getTime() - t < 5: # Simulation Time, not step
    values = trans_field.getSFVec3f()
    print(t, "Pos: {:.2f} {:.2f} {:.2f}".format(*values))
    if supervisor.step(TIME_STEP) == -1:  # controller termination
        print('>> Finish')
        finished = True
        quit()   
    
import sys

# freeze the whole simulation
if finished:
    saveExperimentData()
    sys.exit(0)


# trans_field.setSFVec3f(INITIAL)
# robot.resetPhysics()
# camera.disable()

# image = camera.getImageArray()
# for x in range(0,camera.getWidth()):
  # for y in range(0,camera.getHeight()):
    # red   = image[x][y][0]
    # green = image[x][y][1]
    # blue  = image[x][y][2]
    # gray  = (red + green + blue) / 3
    # print 'r='+str(red)+' g='+str(green)+' b='+str(blue)


