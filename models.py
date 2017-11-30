from tqdm import tqdm_notebook as tqdm
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import os
import time
import math
import torch
import numpy as np
import torchvision
import pandas as pd
import seaborn as sns
import torch.nn as nn
import nibabel as nib
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim.lr_scheduler as schd  
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark=True

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch = batch
        self.lstm = nn.LSTM(input_size = 807, hidden_size = 50,
                            batch_first=True, dropout = 0.5, num_layers = num_layers)
    
    def forward(self, x, seq):
        # Set initial states 
        h0 = Variable(torch.randn(self.num_layers, self.batch, self.hidden_size).float().cuda())
        c0 = Variable(torch.randn(self.num_layers, self.batch, self.hidden_size).float().cuda())
        
        # Forward propagate RNN
        x = pack_padded_sequence(x, seq, batch_first=True)
        out, hidden = self.lstm(x) 
        return out, hidden
    

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch, num_classes):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch = batch
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.5)
        self.fc = nn.Sequential(
            nn.Linear(in_features= 50, out_features= 50),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features= 50, out_features= 3))
 
    def forward(self, x, h0):

        c0 = Variable(torch.zeros(self.num_layers, self.batch, self.hidden_size).float()).cuda()
 
        # Forward propagate RNN
        _, (hidden,cell) = self.lstm(x, (h0, c0))
        out = hidden[-1]
        # Decode hidden state of last time step
        out = self.fc(out)  
        return out



