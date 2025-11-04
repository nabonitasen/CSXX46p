import os
import pickle
import random
import numpy as np
from metrics.metrics_tracker import MetricsTracker
from collections import deque
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def setup(self):
    """
    Setup REINFORCE agent.
    Loads saved model if available, otherwise creates new one.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    model_path = "my-saved-model.pth"
    self.name = "REINFORCE"

    # Hyperparameters
    self.state_size = 291  # Will be determined by state_to_features
    self.action_size = len(ACTIONS)
    self.hidden_size = 128
    self.learning_rate = 1e-4

    self.policy = Policy(self.state_size, self.action_size, self.hidden_size).to(device)
    self.optimizer = optim.Adam(self.policy.parameters(), lr = self.learning_rate)

    # Load saved model if available and not training
    if not self.train and os.path.isfile(model_path):
        self.logger.info("Loading model from saved state.")
        checkpoint = torch.load(model_path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Model loaded from {model_path}")
    else:
        self.logger.info("Setting up model from scratch.")

def act(self, game_state: dict) -> str:
    """
    Choose action using the policy network.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Extract features
    features = state_to_features(game_state)

    if features is None:
        self.logger.warning("Features are None, choosing random action")
        return np.random.choice(ACTIONS)

    # Get action from policy
    if self.train:
        # During training: sample from policy
        action_idx, log_prob = self.policy.act(features)

        # Store log_prob for training (will be used in train.py)
        self.last_log_prob = log_prob
    else:
        # During evaluation: take most probable action (greedy)
        with torch.no_grad():
            action_idx, log_prob = self.policy.act(features)

    return ACTIONS[action_idx]

def state_to_features(game_state: dict) -> np.array:
    board = np.array(game_state["field"]).flatten()
    position = np.array(game_state["self"][3], dtype=np.float32)


    # optional: bombs, coins, others — extend as required

    features = np.concatenate([board.astype(np.float32), position.astype(np.float32)])
    return features