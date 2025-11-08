import numpy as np
import random

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class Maverick(nn.Module):
    '''
    NN with deeper architecture to calculate Q-values
    for 6 actions depending on features (game state)

    Architecture: 23 -> 128 -> 128 -> 64 -> 6
    Improved from: 23 -> 60 -> 6 (too shallow!)
    '''
    def __init__(self):
        super(Maverick, self).__init__()

        self.input_size = 23
        self.number_of_actions = 6

        # LAYERS - Deeper network for better representation learning
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=self.number_of_actions)

        # Dropout for regularization (helps prevent overfitting)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        # Layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Layer 3
        x = F.relu(self.fc3(x))

        # Output layer (no activation - raw Q-values)
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










    