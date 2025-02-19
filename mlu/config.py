################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import os

runs=9
home=os.environ['HOME']

agpy=home+"/agripolis-drl/build/src/agp24"
inputfiles=home+"/agripolis-drl/inputfiles/"
temp_scenario="scenario-temp.txt"

nInvs = 47
epochs = 2000 #3000
simus = 50 #50
topn = 5
QSIZE = simus

