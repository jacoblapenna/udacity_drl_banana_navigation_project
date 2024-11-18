import inspect
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, device, state_size, action_size, fc_size=64, seed=0):
        super(QNetwork, self).__init__()

        self._device = device
        self._action_size = action_size
        self._state_size = state_size
        self._fc_size = fc_size
        
        torch.manual_seed(seed)
 
        self.fc1 = nn.Linear(self._state_size, self._fc_size)
        self.fc2 = nn.Linear(self._fc_size, self._fc_size)
        self.fc3 = nn.Linear(self._fc_size, self._action_size)
    
    def get_greedy_action_from_state(self, state, epsilon):
        """ assumes model is already in eval() mode """
        self.eval()

        if not self.training:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
            with torch.no_grad():
                action_values = self.forward(state)
            
            # Return greedy action with 1 - epsilon probability
            if random.random() > epsilon:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self._action_size))
        else:
            raise Exception("Cannot calculate action in training mode.")
       
        self.train()
      
    def get_norm(self):
        return torch.norm(torch.cat([p.view(-1) for p in self.parameters()]), p=2)
    
    def percent_change(self, initial, final):        
        return (final - initial)/initial * 100
        
       
    def forward(self, state):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))
