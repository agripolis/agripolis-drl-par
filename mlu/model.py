################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import torch
import torch.nn as nn
import torch.nn.functional as F 

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        #print("action: ", action)
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        #x = nn.ReLU6()(x)
        #print("critic: ", x)
        #x = torch.where(x.gt(0), x*20, x*10)
        return x

class Critic2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic2, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.linear4 = nn.Linear(input_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, output_size)


    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        sa = torch.cat([state, action], 1)

        x = F.relu(self.linear1(sa))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        y = F.relu(self.linear4(sa))
        y = F.relu(self.linear5(y))
        y = self.linear6(y)

        return x, y

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(sa))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        return q1

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        #print("a:x ",x)
        #x=x/20.  #1500 #torch.where(x.gt(0), x/1000, x/1500)
        x = torch.tanh(x) 
        #x = nn.ReLU6()(x)
        return (x+1)/2 #+0.25
        #return (x+0.5)/2 #+0.25  for Farm 0
