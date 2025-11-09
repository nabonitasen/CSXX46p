import numpy as np
import random

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class Maverick(nn.Module):
    '''
    NN with one hidden layer to calculate Q-values
    for 6 actions depending on features (game state)
    '''
    def __init__(self):
        super(Maverick, self).__init__()

        self.input_size = 26  # Upgraded from 23 to include 3 new safety features
        self.number_of_actions = 6

        #LAYERS - Lightweight architecture (reduced from 128-128-64)
        # Parameter reduction: ~33k params -> ~5k params (85% reduction)
        self.dense1 = nn.Linear(in_features=self.input_size, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=self.number_of_actions)


    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        out = self.out(x)
        return out

    def initialize_training(self, 
                alpha,
                gamma, 
                epsilon, 
                buffer_size,
                batch_size, 
                loss_function,
                optimizer,
                training_episodes):
        self.gamma = gamma
        self.epsilon_begin = epsilon[0]
        self.epsilon_end = epsilon[1]
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.loss_function = loss_function
        self.training_episodes = training_episodes










    