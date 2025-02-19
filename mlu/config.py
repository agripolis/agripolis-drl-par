################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import os

runs=9
home=os.environ['HOME']+'/agripolis-drl-par/'

agpy=home+"build/src/agp24"
inputfiles=home+"inputfiles/"
temp_scenario="scenario-temp.txt"

nInvs = 47
epochs = 10 #2000 #3000
simus = 10 #50 #50
topn = 2 #5
QSIZE = simus

