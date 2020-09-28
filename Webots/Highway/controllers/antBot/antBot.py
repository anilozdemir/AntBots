# This controller is for robot to move.
# The supervisor can control this robot.

from controller import Robot, Motor, Camera

# INIT #
TIMESTEP = 32
robot = Robot()

camera_front = Camera('camera_front')
camera_front.enable(TIMESTEP*2)

Motors = []
for wheel in ["motor"+str(i) for i in range(1,5)]:
    motor = robot.getMotor(wheel)
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
    Motors.append(motor)

# LOOP #
while robot.step(TIMESTEP) != -1:
    for motor in Motors:
        motor.setVelocity(5000)
