# This controller is for robot to move.
# The supervisor can control this robot.

from controller import Robot, Motor, Camera

# INIT #
TIMESTEP = 32
robot = Robot()

camera_front = Camera('camera_front')
camera_front.enable(TIMESTEP*2)
camera_back = Camera('camera_back')
camera_back.enable(TIMESTEP*2)
camera_top = Camera('camera_top')
camera_top.enable(TIMESTEP*2)
leftMotor  = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# LOOP #
while robot.step(TIMESTEP) != -1:
    leftMotor.setVelocity(1000)
    rightMotor.setVelocity(1000)
    pass

