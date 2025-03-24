################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import subprocess
from config import *
from env import *
import pydata_pb2 as md
from util import *
from datetime import datetime
import multiprocessing as mp
from multiprocessing.managers import BaseManager, BaseProxy
from multiprocessing import Process
import torch.multiprocessing as tmp

from ANET import *

import pickle
import random
import queue
import os
import sys
import numpy as np
from pathlib import Path

start_time = datetime.now()

np.random.seed(2024)
rdir = sys.argv[1]
idx = int(sys.argv[2])

resdir=rdir+"/"+ str (idx) +"/"
Path(resdir).mkdir(parents=True, exist_ok=True)

cf= open(resdir+"best_cum_rew.txt", "at")

min_beta = 0.
max_beta = 2 

max_noise_sigma=0.85
min_noise_sigma=0.10
d_noise_sigma=0.05
d_epocs = 100
sigma_noise=max_noise_sigma

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = Agent(len_in, idx=idx-1)
noise = GaussNoise(sigma=sigma_noise)
batch_size = 8 #32 
best_reward = 0
best_c_reward = 0
min_c_reward = 0


class mManager(BaseManager):
    pass

class mManagerProxy(BaseProxy):
    pass


mManager.register("mAgent", Agent)
mManager.register("mNoise", GaussNoise)
mManager.register("mPQueue", queue.PriorityQueue)

res=queue.PriorityQueue(QSIZE)
zmq_port_base = 12 
#initzmq(zmq_port_base)

# parallelizing, sharing priority queues, memory and agent
def simuls(zmqbaseport, i, nsim, ag, nois, resq): #,  shqueue, shmem, shagent):
  initzmq(zmqbaseport)
  for s in range(nsim):
    closed=False
    resq.put((i,s))

    #tp=subprocess.Popen(["python",  "testmp.py", str(i), str(zmqbaseport)] , stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tp=subprocess.Popen([agpy, "--ZMQ_PORT_BASE=" + str(zmqbaseport), inputfiles ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for x in range(runs):
      if not closed:
         recv_message()
         send_beta(0.5)
         rstr = recv_ec()
         rc = recv_closed()
         if rc>0:
             closed = True
      else:
          recv_ec()

    tp.wait()
    #tpout,tperr = tp.communicate()
    #print(tpout)
    #print(tperr)
  closezmq()
    #print("simu: ", os.getpid())


def outputScaler(e, scaler):
    sf = open(resdir+"c-scaler-"+str(e)+".txt", "wt")
    sf.write(str(scaler.n_features_in_)+"\n")
    sf.write(",".join(map(str,scaler.scale_))+"\n")
    sf.write(",".join(map(str,scaler.data_min_))+"\n")
    sf.close()

def outputModel(e):
    global agent
    global best_reward
    global cf
    agent.saveModel(resdir+"model-"+str(e)+".pth")
    with open(resdir+"state_scaler-"+str(e)+".pkl", "wb") as f:
       pickle.dump(agent.memory.scalers[0], f)
    cf.write(str(e)+"\t"+str(best_reward)+"\n")
    cf.flush()
    
    #outputScaler(e, agent.memory.scalers[0])
    agent.memory.outputAll(resdir+"mtrain-"+str(e)+".dat")

    print(f'-------------------------------> {best_reward}\n')


def testing(e):
    global zmq_port_base
    #print("evaluating ...\n")
    closed=False
    agpt=subprocess.Popen([agpy, "--ZMQ_PORT_BASE=" + str(zmq_port_base), inputfiles, mkscenario(inputfiles+temp_scenario, e) ], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)


    cum_reward = 0.
    for r in range(runs):
        if not closed:
            data=recv_message()
            st = get_state(data)
            #st.append(r)
            #print(st)
            #input("state:...")
            
            st=agent.memory.norm_state(st)[0]
            beta = agent.get_action(st)[0]
            #beta = np.clip(beta, min_beta, max_beta)
            send_beta(beta)

            rew = recv_ec()
            print("beta, rew: ", beta, rew)
            
            rc = recv_closed()
            if rc>0:
                closed=True

        else:
            st=[-1]
            beta=-1
            rew= recv_ec()

        cum_reward += rew

    agpt.wait()
    return cum_reward

def get_ep(e):
    global best_c_reward
    global best_reward
    global min_c_reward
    global zmq_port_base

    max_sig=0.7
    min_sig=0.1
    d_sig=0.1
    d_simu=5
    sig=min_sig

    nproc = 22 # mp.cpu_count()-2
    with mManager() as manager:
      mpAgent = manager.mAgent(len_in, idx=idx-1)
      mpNoise = manager.mNoise(sigma=sigma_noise)

      pss=[]
      qss = []
      for i in range(0,2*nproc,2):
          # print(i)
          q = manager.mPQueue(int(simus/nproc))
          p=Process(target=simuls, args=(i,i,int(simus/nproc), mpAgent, mpNoise,q))
          pss.append(p)
          qss.append(q)
      
      for p in pss:
          p.start()

      for x in pss:
          x.join()

      for y in qss:
          t=y.get()
          print(y.qsize(), t[0], t[1])

     # input("hallo")
#     for s in range(simus):
#         if (s+1) % d_simu == 0:
#             sig += d_sig
#             sig = min(sig, max_sig)
#             noise.set_sigma(sig)
# 
#         closed=False
#         #subprocess.run(agripolis, iniputfiles])
#         agp=subprocess.Popen([agpy, "--ZMQ_PORT_BASE=" + str(zmq_port_base), inputfiles ], 
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True)
# 
#         ep=[]
#         eprew=0
#         for r in range(runs):
#             if not closed:
#                 data=recv_message()
#                 st = get_state(data)
#                 
#                 if e > 0:
#                    stn=agent.memory.norm_state(st)[0]
#                    beta = agent.get_action(stn)[0]
#                 else:
#                    beta = agent.get_action(st)[0]
#                 #beta = np.clip(beta, min_beta, max_beta)
#                 delta = noise()
#                 #print(delta)
#                 beta += delta[0]
#                 beta = np.clip(beta, min_beta, max_beta)
#                 send_beta(beta)
# 
#                 rew = recv_ec()
#                 #print("beta, rew: ", beta, rew)
#                 
#                 rc = recv_closed()
#                 if rc>0:
#                     closed=True
# 
#             else:
#                 st=[-1]
#                 beta=-1
#                 rew= recv_ec()
# 
# 
#             ep.append((st, [beta],  [rew]))
#             eprew += rew
# 
#         res.put((-eprew, ep))
#         agp.wait()
#         #agpout, agperr = agp.communicate()
#         #print(agpout)
#         #print(agperr)
# 
# 
    #print(res.queue)

    #TODO
    # merge res-i

#     x=0
#     while x < topn  and not res.empty():
#        t=res.get()
#        ep= t[1]
#        c_rew = -t[0]
#        if len(agent.memory) < agent.memory_size or c_rew > min_c_reward:
#           agent.memory.rpush_episode(t) 
#           min_c_reward = agent.memory.get_min_reward()
#           #print(c_rew, min_c_reward)
#        if c_rew > best_c_reward:
#            best_c_reward = c_rew
#        x+=1 
# 
#     #print("mem_size: ", len(agent.memory), best_c_reward)
#     res.queue.clear()
# 
#     #print("len: ", len(agent.memory), "\tBS: ", batch_size)
#     if len(agent.memory) > batch_size:
#         agent.memory.norm()
#         for i in range(topn):
#             agent.update(batch_size)  #learning()
#             
#     if e % 1 == 0:
#         testreward = testing(e)
#         #print(f'--test--: {testreward}')
#         if testreward > best_reward:
#             best_reward = testreward 
#             outputModel(e)
#         else:
#             print(testreward,'\n')
# 
outputMemory = True
LenMem = 10000
curlen  = 0

if __name__ == '__main__':
   mp.set_start_method('spawn')

   for e in range(epochs):
       #if (e+1) % d_epocs == 0:
       #    sigma_noise -= d_noise_sigma
       #    noise.set_sigma(max(min_noise_sigma, sigma_noise)) 
       #    #print(sigma_noise)
   
       noise.reset()
       print(f"======= epoch {e} =========") 
       get_ep(e)
   
       buflen = len(agent.memory)
   
   cf.close()
   
   #closezmq()
   
   delt_time = datetime.now() - start_time
   print("total time in ms: ", delt_time.total_seconds()*10**3)
