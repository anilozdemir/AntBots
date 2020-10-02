import os
import numpy as np
import torch
from torch.nn import functional as F

from torch.utils.data import DataLoader, ConcatDataset
import pyRC.network as RC
from pyRC.analyse.perfectMemory import *
import pyRC.analyse.utils as utA
from pyRC.datasets.utils import ImageDataset
from pytorch_lightning.core.lightning import LightningModule


#TODO Move into pyRC.datasets
class antBotDatasets():
    def __init__(self, expPath, dataSetNames):
        self.expPath     = expPath # experiment path
        # TODO put this is as mode='auto', or operate with given list of string
        # self.strImages   = sorted([f.replace('.npy','') for f in os.listdir(self.expPath)])# get the name of all data in the given experiment path
        self.strImages   = dataSetNames
        self.allImages   = np.stack([np.load(self.expPath+f+'.npy') for f in self.strImages]) # load all the data matched in strImages
        self.nDatasets   = self.allImages.shape[0]
        self.nImages     = self.allImages.shape[1]
        self.camSize     = self.allImages.shape[-2:]
        print(f'>> Loaded {self.nDatasets} datasets of {self.nImages} images each!')
        
    def get(self, strDataset, nImages, h, w):
        ''' Note:
            dataSet in shape: torch.Size([num_images, height, width, batchSize=1]) # batchSize=1 reserved for batch, to be permuted to top
        '''
        dsID    = self.strImages.index(strDataset)
        Images0 = self.allImages[dsID]
        Images  = np.stack([IMAGEOP(img,h,w) for img in Images0]) # need to swap axis of height(2) and width(1)
        ImagesT = torch.Tensor(Images).permute(0,2,1).unsqueeze(-1) # add dimension for batch operations
        iLabels = range(self.nImages)
        self.dataSet           = ImageDataset(ImagesT, iLabels)
        self.dataSet.name      = strDataset
        dataLoader             = DataLoader(self.dataSet, batch_size = 1, shuffle = False)
        dataLoader.nInput      = h*w   # image height is the number of inputs
        dataLoader.nOutput     = nImages # nImages to choose from
        dataLoader.name        = strDataset
        return dataLoader

    def concatDatasets(self, dataSetList):
        dataSet    = ConcatDataset(dataSetList)
        dataLoader = DataLoader(dataSet, batch_size = 1, shuffle = False)
        return dataLoader


class antBotEMB(LightningModule):
    def __init__(self, config): #, trainDL, testDL):
        super().__init__()
        self.hparams = {**config['hyperparameters'], **config['experimentParameters'], **config['modelParameters'], **config['experimentDetails']} # unzip nested dict
        self.dlargs  = {'nImages': self.hparams['nImages'], 'h': self.hparams['height'], 'w': self.hparams['width']}
        self.RC      = RC.ESN_NA(self.hparams, readoutType = self.hparams['readoutType'])
        print('>> Network is constructed!')

    def forward(self, x):
        return self.RC(x)

    def prepare_data(self):
        antBotData = antBotDatasets(self.hparams['dataSetPath'], self.hparams['dataSets'])
        self.DL, self.DS = [], []
        for dataSetName in self.hparams['dataSets']:
            DL = antBotData.get(dataSetName, **self.dlargs)
            DS = antBotData.dataSet
            self.DL.append(DL)
            self.DS.append(DS)
        
        # self.DL0 = antBotData.get(self.dataStr[0], **self.dlargs)
        # DS0      = antBotData.dataSet # get the dataset of DL0 
        # self.DL1 = antBotData.get(self.dataStr[1], **self.dlargs)
        # DS1      = antBotData.dataSet # get the dataset of DL1
        # self.DL2 = antBotData.get(self.dataStr[2], **self.dlargs)
        # DS2      = antBotData.dataSet # get the dataset of DL2 

        self.DLT = antBotData.concatDatasets(self.DS) # train DL (DLT) is concat of DL0, DL1, DL2
        print('>> Datasets are loaded!')
        
        self.hparams['nParams'] = utA.modelParameters(self.RC, returnOnly=True).item()
        
    def train_dataloader(self):
        return self.DLT
      
    def test_dataloader(self):
        # return [self.DL0, self.DL1, self.DL2]
        return self.DL
    
    def configure_optimizers(self):
        params = [{'params': self.RC.Wout, 'lr': self.hparams['learningRate']}] 
        return torch.optim.Adam(params)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:  # At the beginning of each epoch, reset the model! 
            self.RC.reset()
            
        x, y  = batch
        y_hat = self(x)
        loss  = F.cross_entropy(y_hat, y)
        logs_loss = {'train_loss': loss}
        return {'loss': loss, 'log': logs_loss}

    def test_step(self, batch, batch_idx, dataloader_idx): 
        x, y    = batch
        y_hat   = self(x)
        diff    = torch.abs(y - torch.argmax(y_hat))
        return {'diff': diff, 'y_pred': y_hat}

    def test_epoch_end(self, outputs):
        logs_err = {}
        logs_acc = {}
        for i, output in enumerate(outputs):
            err = torch.stack([x['diff'] for x in output]).float().mean()
            acc = torch.stack([x['diff']<self.hparams['tolerance'] for x in output]).float().mean()*100 # percentage
            logs_err['err'+str(i)] = err
            logs_acc['acc'+str(i)] = acc

        # err_0 = torch.stack([x['diff'] for x in outputs[0]]).float().mean()
        # err_1 = torch.stack([x['diff'] for x in outputs[1]]).float().mean()
        # err_2 = torch.stack([x['diff'] for x in outputs[2]]).float().mean()
        
        # acc_0 = torch.stack([x['diff']<self.hparams['tolerance'] for x in outputs[0]]).float().mean()*100 # percentage
        # acc_1 = torch.stack([x['diff']<self.hparams['tolerance'] for x in outputs[1]]).float().mean()*100 # percentage
        # acc_2 = torch.stack([x['diff']<self.hparams['tolerance'] for x in outputs[2]]).float().mean()*100 # percentage
        
        # logs_err = {'err_0': err_0, 'err_1': err_1, 'err_2': err_2}
        # logs_acc = {'acc_0' : acc_0,  'acc_1' : acc_1 , 'acc_2': acc_2}
        return {'log': {**logs_err, **logs_acc}}