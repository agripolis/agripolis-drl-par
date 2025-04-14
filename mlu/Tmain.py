################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import subprocess
from config import *
from Tenv import *
import pydata_pb2 as md
from util import *
from datetime import datetime

from threading import Thread

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
testf = open(resdir+"testing.txt", "at")

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


zmq_port_base=0

res=queue.PriorityQueue(QSIZE)

# parallelizing with threading
def simuls(e, i, nsim, ag, sig, resq):
  noi = GaussNoise(sigma=sig)
  for s in range(nsim):
    closed=False

    #tp=subprocess.Popen(["python",  "testmp.py", str(i), str(zmqbaseport)] , stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tp=subprocess.Popen([agpy, "--ZMQ_PORT_BASE=" + str(i), inputfiles ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    ep=[]
    eprew=0
    for r in range(runs):
        if not closed:
            data=recv_message(i)
            st = get_state(data)
            
            if e > 0:
               stn=ag.memory.norm_state(st)[0]
               beta = ag.get_action(stn)[0]
            else:
               beta = ag.get_action(st)[0]

            delta = noi()
            #print(delta)
            beta += delta[0]
            beta = np.clip(beta, min_beta, max_beta)
            send_beta(beta,i)
    
            rew = recv_ec(i)
            #print("beta, rew: ", beta, rew)
            
            rc = recv_closed(i)
            if rc>0:
                closed=True
    
        else:
            st=[-1]
            beta=-1
            rew= recv_ec(i)
    
    
        ep.append((st, [beta],  [rew]))
        eprew += rew
 
    resq.put((-eprew, ep))

    tp.wait()
    #tpout,tperr = tp.communicate()
    #print(tpout)
    #print(tperr)
    #print("simu: ", os.getpid())




def outputScaler(e, scaler):
    sf = open(resdir+"c-scaler-"+str(e)+".txt", "wt")
    sf.write(str(scaler.n_features_in_)+"\n")
    sf.write(",".join(map(str,scaler.scale_))+"\n")
    sf.write(",".join(map(str,scaler.data_min_))+"\n")
    sf.close()

def outputModel(e):
    agent.saveModel(resdir+"model-"+str(e)+".pth")
    with open(resdir+"state_scaler-"+str(e)+".pkl", "wb") as f:
       pickle.dump(agent.memory.scalers[0], f)
    cf.write(str(e)+"\t"+str(best_reward)+"\n")
    cf.flush()
    
    #outputScaler(e, agent.memory.scalers[0])
    agent.memory.outputAll(resdir+"mtrain-"+str(e)+".dat")

    #print(f'-------------------------------> {best_reward}\n')


def testing(e):
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
            #print("beta, rew: ", beta, rew)
            testf.write(f"beta, rew: {beta}, {rew}\n")
            
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

    max_sig=0.51
    min_sig=0.01
    d_sig=0.1
    d_simu=5
    sig=min_sig


    pss = []
    qss = []
    t1=datetime.now()
    for i in range(0,2*nthread,2):
        # print(i)
        sig=min_sig+ i*(max_sig-min_sig)/nthread
        q = queue.PriorityQueue(int(simus/nthread))
        p=Thread(target=simuls, args=(e, i,int(simus/nthread), agent, sig, q))
        pss.append(p)
        qss.append(q)
    
    for p in pss:
        p.start()

    for x in pss:
        x.join()

    #print("T: ", datetime.now()-t1)
    # merge queues
    for y in qss:
        while not y.empty():
            res.put(y.get()) 


    x=0
    while x < topn  and not res.empty():
       t=res.get()
       ep= t[1]
       c_rew = -t[0]
       if len(agent.memory) < agent.memory_size or c_rew > min_c_reward:
          agent.memory.rpush_episode(t) 
          min_c_reward = agent.memory.get_min_reward()
          #print(c_rew, min_c_reward)
       if c_rew > best_c_reward:
           best_c_reward = c_rew
       x+=1 
 
    #print("mem_size: ", len(agent.memory), best_c_reward)
    res.queue.clear()
 
    #print("len: ", len(agent.memory), "\tBS: ", batch_size)
    if len(agent.memory) > batch_size:
        agent.memory.norm()
        for i in range(topn):
            agent.update(batch_size)  #learning()
            
    if e % 1 == 0:
        testreward = testing(e)
        #print(f'--test--: {testreward}')
        if testreward > best_reward:
            best_reward = testreward 
            outputModel(e)
            testf.write(f'-------------------------------> {best_reward}\n\n')
            
        else:
            #print(testreward,'\n')
            testf.write(str(testreward)+'\n\n')

outputMemory = True
LenMem = 10000
curlen  = 0

if __name__ == '__main__':
   nthread = 45 

   initzmq0(nthread)

   for e in range(epochs):
       #if (e+1) % d_epocs == 0:
       #    sigma_noise -= d_noise_sigma
       #    noise.set_sigma(max(min_noise_sigma, sigma_noise)) 
       #    #print(sigma_noise)
   
       noise.reset()
       #print(f"======= epoch {e} =========") 
       testf.write(f'======= epoch {e} =========\n') 
       get_ep(e)
   
       buflen = len(agent.memory)
   
   cf.close()
   closezmq0(nthread)
   
   delt_time = datetime.now() - start_time
   testf.write("total time in ms: "+ str(delt_time.total_seconds()*10**3)+ "\n")
   testf.close() 
   print("total time in ms: ", delt_time.total_seconds()*10**3)
