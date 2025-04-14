################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import os

runs=9
home=os.environ['HOME']+'/agripolis-drl-par/'

agpy=home+"build/src/agp24"
#inputfiles=home+"inputfiles/"
inputfiles=home+"inputfiles-large/"
temp_scenario="scenario-temp.txt"

nInvs = 47
epochs = 1000 #2000 #3000
simus = 180 #135 #450 #50 #50
topn = 20  #30 #5
QSIZE = simus

