import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py
import pickle,sys,time

class GateMechanism(nn.Module):
    '''
    Gating Mechanism is used to filter out the noise
    information of vision and acoustic modality
    '''
    def __init__(self, inputx_dim):
        super(GateMechanism, self).__init__()
        self.d_x = inputx_dim
        self.d_out = inputx_dim
        self.MLP = nn.Sequential(
            nn.Linear(self.d_x, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, self.d_out),
            nn.Sigmoid()
        )

    def __call__(self, inputx):

        return self.forward(inputx)

    def forward(self, inputx):
        gate = self.MLP(inputx)
        output = torch.mul(inputx, gate)

        return output









