################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import os

runs=9
home=os.environ['HOME']+'/ALTMARK/agripolis-drl-par/'

agpy=home+"build/src/agp24"
#inputfiles=home+"inputfiles/"
inputfiles=home+"inputfiles-large/"
temp_scenario="scenario-temp.txt"

nInvs = 58  #47
epochs = 300 # 300 #2000 #3000
simus = 270 #270  #180 #135 #450 #50 #50
topn = 30 #30  #30 #5
QSIZE = simus

