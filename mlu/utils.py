################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import numpy as np
import gymnasium as gym
from collections import deque
import random
import heapq

#random.seed("agripolis2020")
#np.random.seed(2023)
from sklearn.preprocessing import MinMaxScaler

# Ornstein-Uhlenbeck Noise
class OUNoise:
    def __init__(self, mu=0.0, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(self.mu) #size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class GaussNoise:
    def __init__(self, mu=0.0, sigma=0.8, len=1):
        self.mu = mu
        self.sigma = sigma
        self.len = len

    def set_sigma(self, s):
        self.sigma  = s

    def reset(self):
        pass

    def __call__(self):
        x = np.random.normal(self.mu, self.sigma, self.len)
        return 0.5*x

    def __repr__(self):
        return 'GaussianNoise(mu={}, sigma={}, len={})'.format(self.mu, self.sigma, self.len)


class Memory:
    LEN=5
    def __init__(self, max_size):
        self.max_size = max_size
        self.rheap = []
        #self.buffer = deque(maxlen=max_size)
        self.nbuffer = deque(maxlen=max_size)
        self.scalers=[MinMaxScaler() for i in range(self.LEN)]
        self.scalers[1].data_min_ = 0
        self.scalers[1].data_max_ = 1.0
        self.scalers[1].data_range_ = 1.0
        self.scalers[1].min_ = 0
        self.scalers[1].scale_ = 1.0
   
    def norm_state(self, s):
        #print(s)
        s=[s]
        #print(self.scalers[0].transform(s))
        #input("state nstate")
        return self.scalers[0].transform(s)

    def get_min_reward(self):
        return min(self.rheap)[0]

    def rpush(self, state, action, reward, next_state, done, crew):
        experience = (state, action, reward, next_state, done)
        if len(self.rheap) >= self.max_size:
            heapq.heappop(self.rheap)
        heapq.heappush(self.rheap, (crew, experience))


    def rpush_episode(self, teps):
        cr = -teps[0]
        eps = teps[1]
        cnt = len(eps)
        #print("len, cr: ", cnt, cr)
        #input("CR:")
        #av_r = cr/cnt
        T = [0]
        s0, a0, r0 = eps[0]
        #s0, a0, _ = eps[0]
        #r0=[av_r]
        #r=[av_r]
        for c in range(1,cnt):
            s, a, r = eps[c]
            #s, a, _ = eps[c]
            if s == [-1]:
                s = s0
                self.rpush(s0, a0, r0, s, [0], cr)
                break
            if a0[0] > -0.3:
                self.rpush(s0, a0, r0, s, [0], cr)
            s0, a0, r0 = s, a, r
        self.rpush(s0, a0, r0, s0, [1], cr)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def push_episode(self, eps):
        cnt = len(eps)
        T = [0]
        s0, a0, r0 = eps[0]
        for c in range(1,cnt):
            s, a, r = eps[c]
            if s == [-1]:
                break
            if a0[0] > -0.3:
                self.push(s0, a0, r0, s, [0])
            s0, a0, r0 = s, a, r
        #self.push(s0, a0, r0, T, 1)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.nbuffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.rheap)

    def outputAll(self, fname):
        f= open(fname, "wt")
        f.write(f'state, action, cum_reward\n')

        for crew, d in self.rheap:
            (s, [a], _, _, _) = d
            f.write(f'{str(s)}, {str(a)}, {str(crew)}\n') 
        f.close()

    def outputf(self, fname, cnt):
        f= open(fname, "wt")
        f.write(f'state, action, reward, next_state, done\n')

        batch = random.sample(self.rheap, cnt)
        for _, d in batch:
            (s, [a], [r], s2, [df]) = d
            f.write(f'{str(s)}, {str(a)}, {str(r)}, {str(s2)}, {str(df)}\n') 
        f.close()
    
    def routputf(self, fname, cnt):
        f= open(fname, "wt")
        #f.write(f'state, action, reward, next_state, done\n')
        f.write(f'normalized reward\n')

        inds = [*range(cnt)]
        bnbuffer = [self.nbuffer[i] for i in inds]
        for _, _, r, _, _  in bnbuffer:
            f.write(f'{r}\n')

        f.close()

    def norm(self):
        d=[[x[i] for _, x in self.rheap] for i in range(self.LEN)]
        for i in range(self.LEN):
            if i != 1: #action
              self.scalers[i].fit(d[i])
       
        self.nbuffer = [ [self.scalers[i].transform([x[i]]).tolist()[0]  
              for i in range(self.LEN)] for _, x in self.rheap]
