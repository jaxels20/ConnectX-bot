from typing import Any, Mapping
import torch.nn as nn
import torch
import pickle
import base64
import torch.nn.functional as F

class ConnectFourNet(nn.Module):
    def __init__(self):
        super(ConnectFourNet, self).__init__()
        self.fc1 = nn.Linear(6*7, 256)  # Assuming the board is represented as a 6x7 matrix
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        
        # Policy head: Probabilities for each of the 7 possible columns
        self.fc6_policy = nn.Linear(64, 7)
        # Value head: Scalar prediction of game outcome (-1 to 1)
        self.fc6_value = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the board
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        policy_logits = self.fc6_policy(x)
        policy_probs = F.softmax(policy_logits, dim=1)
        
        value = torch.tanh(self.fc6_value(x))
        
        return policy_probs, value

    
    def save(self, path='model.pth'):
        torch.save(self.state_dict(), path)
    
    def serialize(self, path='state_dict.txt'):
        state_dict = self.state_dict()
        state_dict_bytes = pickle.dumps(state_dict)
        state_dict_str = base64.b64encode(state_dict_bytes).decode('utf-8')
        
        with open(path, 'w') as f:
            f.write(state_dict_str)
    
    def load_state_dict_from_file(self, path="state_dict.txt"):
        with open(path, 'r') as f:
            state_dict_str = f.read()
        
        state_dict_bytes = base64.b64decode(state_dict_str)
        state_dict = pickle.loads(state_dict_bytes)
        
        self.load_state_dict(state_dict)