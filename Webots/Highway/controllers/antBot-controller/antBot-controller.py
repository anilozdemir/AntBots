# antBot-controller, controlled by supervisor
from controller import Robot
import numpy as np
import sys

class antBot(Robot):
    timeStep = 32
    
    def __init__(self):
        super(antBot, self).__init__()
        self.camera_front = self.getCamera('camera_front')
        self.camera_front.enable(self.timeStep)    
        self.camera_spherical = self.getCamera('camera_spherical')
        self.camera_spherical.enable(self.timeStep) 
           
        self.receiver = self.getReceiver('receiver')
        self.receiver.enable(self.timeStep)
        self.Images = []
        
    def run(self):
        while True:
            if self.receiver.getQueueLength() > 0:
                message = self.receiver.getData().decode('utf-8')
                self.receiver.nextPacket()
                # print(message)
                if message == 'CAPTURE':
                    image = np.array(self.camera_front.getImageArray())
                    self.Images.append(image)
                elif message == 'FINISH':
                    break
            # Perform a simulation step, quit the loop when
            # Webots is about to quit.
            if self.step(self.timeStep) == -1:
                break

    def save(self, fileName):
        np.save(fileName ,np.stack(self.Images))
        print(f'>> {fileName}.npy is saved!')
        # freeze the whole simulation
        sys.exit(0)

agent = antBot()
agent.run()
# agent.save('Yo2')