################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from model import *
from utils import *
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def manualseed(i):
    seeds = [2020, -2023, -2020, 2023, -1]
    seed = seeds[i] 
    torch.manual_seed(seed)
    #pass

class Agent:
    def __init__(self, num_states, hidden_size=16, actor_learning_rate=1e-4, critic_learning_rate=1e-5, 
gamma=0.8, tau=1e-2, max_memory_size=200, idx=1): #0.99  #100000
        # Params
        self.num_states = num_states
        self.num_actions = 1
        self.gamma = gamma
        self.tau = tau
        self.idx = idx
        self.memory_size=max_memory_size

        # Networks
        manualseed(self.idx)
        self.actor = Actor(self.num_states, hidden_size, self.num_actions).to(device)
        manualseed(self.idx)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(device)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.loss = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

    
    def get_action(self, state):
        state = np.array(state)
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
        action = self.actor.forward(state)
        #action = action.detach().numpy()[0,0]
        action = action.cpu().data.numpy().flatten()
        #print("action: ", action)
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        done = torch.FloatTensor(done).to(device)
    
        # Actor loss
        n_actions = self.actor.forward(states)
        #print(actions)
        #print(n_actions)
        policy_loss = self.loss(n_actions, actions)
        #print(f'actor_loss: {policy_loss}')
        #input("hhh")
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
    
        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
    def saveModel(self, filepath):
        torch.save(self.actor, filepath)

    def loadModel(self, filepath):
        return torch.load(filepath)
