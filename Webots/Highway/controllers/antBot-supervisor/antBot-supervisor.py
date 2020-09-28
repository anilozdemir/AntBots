# antBot-supervisor
from controller import Supervisor
import sys

class antBotDriver(Supervisor):
    timeStep = 32
    
    def __init__(self):
        super(antBotDriver, self).__init__()
        self.robot  = self.getFromDef("antBot")
        self.trans  = self.robot.getField("translation")
        self.values = self.trans.getSFVec3f()
        self.emitter = self.getEmitter('emitter') 
        # TODO: send data for initilisation to antBot

    def run(self, LIST):
        # Main loop:
        XY = [-7, 0.5]
        INIT = XY + [0,]
        self.trans.setSFVec3f(INIT)
        print('>> supervisor started')
        for z in LIST:
            message = 'CAPTURE'.encode('utf-8')
            self.emitter.send(message)
            values = self.trans.getSFVec3f()
            POS = XY +[z,]
            self.trans.setSFVec3f(POS)
            if self.step(self.timeStep) == -1:  # controller termination
                pass    
        message = 'FINISH'.encode('utf-8')
        self.emitter.send(message)  
        sys.exit(0) # freeze the whole simulation


driver = antBotDriver()
driver.run(range(0,200,1))