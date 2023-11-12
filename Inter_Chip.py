import gurobipy as gp
import yaml
import pandas as pd
import pydot
import math
import copy
import argparse
import pprint
import sys
import numpy as np

    
    
    
    
    
    
model = gp.Model()
model.params.NonConvex = 2
model.Params.Threads = 120

Nc = 4
layer = 96
M = [36864, 12288, 49152, 12288]
K = [12288, 12288, 12288, 49152]
N = [2048, 2048, 2048, 2048]
datatype = 2
num_chip = 8
    



    
batch = model.addMVar(1, name='batch', vtype=gp.GRB.INTEGER, lb=1)
tp = model.addMVar(1, name='tp', vtype=gp.GRB.INTEGER, lb=1)
pp = model.addMVar(1, name='pp', vtype=gp.GRB.INTEGER, lb=1)
num_layer_per_pp = model.addMVar(1, name='num_layer_per_pp', vtype=gp.GRB.INTEGER, lb=1)

    

m_shard = model.addMVar(Nc, name='m_shard', vtype=gp.GRB.BINARY)
k_shard = model.addMVar(Nc, name='k_shard', vtype=gp.GRB.BINARY)

per_kernel_network_bytes = model.addMVar(Nc, name='per_kernel_network_bytes', vtype=gp.GRB.CONTINUOUS, lb=0)
per_kernel_memory_bytes = model.addMVar(Nc, name='per_kernel_memory_bytes', vtype=gp.GRB.CONTINUOUS, lb=0)

network_bytes = model.addMVar(1, name='network_bytes', vtype=gp.GRB.CONTINUOUS, lb=0)
memory_bytes = model.addMVar(1, name='memory_bytes', vtype=gp.GRB.CONTINUOUS, lb=0)




model.addConstr(tp[0] @ pp[0] == num_chip)
model.addConstr(pp[0] @ num_layer_per_pp[0] == layer)



for i in range(Nc):
    model.addConstr(m_shard[i] + k_shard[i] == 1)


for i in range(Nc):

    next_one = (i+1) % Nc
    
    mm = model.addVar(vtype=gp.GRB.BINARY)   
    mk = model.addVar(vtype=gp.GRB.BINARY)
    
    km = model.addVar(vtype=gp.GRB.BINARY)
    kk = model.addVar(vtype=gp.GRB.BINARY)

    
    model.addConstr(mm == gp.and_(m_shard[i].tolist()[0], m_shard[next_one].tolist()[0]))
    model.addConstr(mk == gp.and_(m_shard[i].tolist()[0], k_shard[next_one].tolist()[0]))
    
    model.addConstr(km == gp.and_(k_shard[i].tolist()[0], m_shard[next_one].tolist()[0]))
    model.addConstr(kk == gp.and_(k_shard[i].tolist()[0], k_shard[next_one].tolist()[0]))
    
    
    
    
    
    model.addConstr((mm == 1) >> (per_kernel_network_bytes[i].tolist()[0] == M[i] * N[i]))
    model.addConstr((mk == 1) >> (per_kernel_network_bytes[i].tolist()[0] == 0))
    
    model.addConstr((km == 1) >> (per_kernel_network_bytes[i].tolist()[0] == M[i] * N[i]))
    model.addConstr((kk == 1) >> (per_kernel_network_bytes[i].tolist()[0] == M[i] * N[i]))
    

    
    
    tmp1 = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=0)
    tmp2 = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=0)
    tmp3 = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=0)
    tmp4 = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=0)
    
    
    model.addConstr(tmp1[0] @ tp == K[i])
    model.addConstr(tmp2[0] @ tp == batch)
    
    
    model.addConstr(tmp3[0] == M[i] * tmp1[0] + K[i] * N[i] * batch[0] + M[i] * N[i] * tmp2[0])
    model.addConstr(tmp4[0] == M[i] * tmp1[0] + K[i] * N[i] * tmp2[0] + M[i] * N[i] * batch[0])

    
    
    model.addConstr((m_shard[i].tolist()[0] == 1) >> (per_kernel_memory_bytes[i].tolist()[0] == tmp3[0].tolist()[0]))
    model.addConstr((k_shard[i].tolist()[0] == 1) >> (per_kernel_memory_bytes[i].tolist()[0] == tmp4[0].tolist()[0]))




tmp5 = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=0)  
model.addConstr(tmp5[0] == per_kernel_memory_bytes @ np.ones(Nc))



model.addConstr(memory_bytes[0] == tmp5[0] @ num_layer_per_pp[0] * datatype)
model.addConstr(memory_bytes[0] <= 1.5*1024**4)


model.addConstr(network_bytes[0] == per_kernel_network_bytes @ np.ones(Nc) * datatype)


model.setObjectiveN(network_bytes[0], 0, 1, gp.GRB.MINIMIZE)
model.setObjectiveN(memory_bytes[0], 1, 0, gp.GRB.MAXIMIZE)
model.optimize()




for v in model.getVars():
    print(v.varName, v.x)




