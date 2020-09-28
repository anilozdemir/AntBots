# This controller is for robot to move.
# The supervisor can control this robot.

from controller import Robot, Motor, Camera, GPS
import numpy as np

# INIT #
TIMESTEP = 32
robot = Robot()

camera_front = Camera('camera_front')
camera_front.enable(TIMESTEP)

gps = GPS('gps')
gps.enable(TIMESTEP)

Motors = []
for wheel in ["motor"+str(i) for i in range(1,5)]:
    motor = robot.getMotor(wheel)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    Motors.append(motor)

# print(dir(robot))
# LOOP #
counter = 0
while robot.step(TIMESTEP) != -1:
    image = camera_front.getImageArray()
    values = gps.getValues()
    ## print("{}-Pos: {:.2f} {:.2f} {:.2f}".format(counter,*values))
    counter += 1
    for motor in Motors:
        motor.setVelocity(5000)
