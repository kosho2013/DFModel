import gurobipy as gp
import argparse
import numpy as np
import setup_pb2
import pprint
from enum import Enum
from google.protobuf import text_format
import pydot
import copy

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='pls pass in the folder name under the current directory (named after the DL model)', required=True)
args = parser.parse_args()
name = args.name


# read in pd file
dse = setup_pb2.DSE()
with open('./'+name+'/'+'dse_sharded.pb', "rb") as file:
    dse.ParseFromString(file.read())
    
    








# get kernels
kernel_name = []
for kernel in dse.dataflow_graph.kernels:
    kernel_name.append(kernel.name)

kernel_type = []
for kernel in dse.dataflow_graph.kernels:
    kernel_type.append(kernel.batch_gemm_elementwise_outer_m_k_m.type)

outer = []
for kernel in dse.dataflow_graph.kernels:
    outer.append(kernel.batch_gemm_elementwise_outer_m_k_m.outer)

M = []
for kernel in dse.dataflow_graph.kernels:
    M.append(kernel.batch_gemm_elementwise_outer_m_k_m.M)

K = []
for kernel in dse.dataflow_graph.kernels:
    K.append(kernel.batch_gemm_elementwise_outer_m_k_m.K)

N = []
for kernel in dse.dataflow_graph.kernels:
    N.append(kernel.batch_gemm_elementwise_outer_m_k_m.N)

sharding = []
for kernel in dse.dataflow_graph.kernels:
    sharding.append(kernel.batch_gemm_elementwise_outer_m_k_m.N)

node_dict = {}
i = 0
for kernel in dse.dataflow_graph.kernels:
    node_dict[kernel.id] = i
    i += 1

num_kernel = len(kernel_name)

seq_len = dse.training.seq_len




# get buffers
startIdx = []
endIdx = []
for connection in dse.dataflow_graph.connections:
    startIdx.append(connection.startIdx)
    endIdx.append(connection.endIdx)

depth = []
for connection in dse.dataflow_graph.connections:
    depth.append(connection.buffer_depth)

tensor_size = []
for connection in dse.dataflow_graph.connections:
    tensor_size.append(connection.tensor_size)


edge_dict = {}
i = 0
for connection in dse.dataflow_graph.connections:
    edge_dict[(connection.startIdx, connection.endIdx)] = i
    i += 1

num_edge = len(startIdx)
    





# get system info
class BasicTopology(Enum):
    NO_BASICTOPOLOGY = 0
    R = 1
    FC = 2
    SW = 3



Core = dse.system.accelerator.core
SRAM_Cap = dse.system.accelerator.sram_cap
VecWidth = dse.system.accelerator.systolic_width
StageWidth = dse.system.accelerator.systolic_width
Freq = dse.system.accelerator.freq
DRAM_BW = dse.system.accelerator.dram_bw
DRAM_Cap = dse.system.accelerator.dram_cap



if dse.system.topo == 1: # 2D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]

elif dse.system.topo == 1: # Dragonfly
    topology = [BasicTopology.FC.value, BasicTopology.FC.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]

elif dse.system.topo == 1: # DGX-1
    topology = [BasicTopology.FC.value, BasicTopology.SW.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]

elif dse.system.topo == 1: # DGX-2
    topology = [BasicTopology.SW.value, BasicTopology.SW.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]

else: # 3D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y, dse.system.link_bw_z]





C = num_kernel


model = gp.Model()
model.params.NonConvex = 2
model.Params.Threads = 10
model.params.MIPGap = 0.1    # 10%
model.params.TimeLimit = 300  # 5 minutes
model.optimize()



# sharding
tile_size = model.addVar(name='tile_size', vtype=gp.GRB.INTEGER, lb=0)
num_tile = model.addVar(name='num_tile', vtype=gp.GRB.INTEGER, lb=0)
model.addConstr(tile_size * num_tile == seq_len)

sharded_intermediate_buffer_size = model.addMVar(num_edge, name='sharded_intermediate_buffer_size', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(num_edge):
    model.addConstr(sharded_intermediate_buffer_size[i] * num_tile == tensor_size[i])





shard_M = model.addVar(num_kernel, name='shard_M', vtype=gp.GRB.INTEGER, lb=0)
shard_K = model.addVar(num_kernel, name='shard_K', vtype=gp.GRB.INTEGER, lb=0)
shard_N = model.addVar(num_kernel, name='shard_N', vtype=gp.GRB.INTEGER, lb=0)



for i in range(num_kernel):



Config = model.addMVar(num_kernel, name='Config', vtype=gp.GRB.INTEGER, lb=0)
Ab_onchip = model.addMVar((num_edge, C), name='Ab_onchip', vtype=gp.GRB.BINARY) # on-chip
Ab_dram = model.addMVar((num_edge, C), name='Ab_dram', vtype=gp.GRB.BINARY) # to/from DRAM
Ac = model.addMVar((num_kernel, C), name='Ac', vtype=gp.GRB.BINARY)
Ad = model.addMVar((Nd, C), name='Ad', vtype=gp.GRB.BINARY)





# compute assignment   
t1 = np.ones((C))
for i in range(num_kernel):
    model.addConstr(Ac[i, :] @ t1 == 1)
    
    
t2 = np.zeros((C))
for i in range(C):
    t2[i] = i
for i in range(num_kernel):
    model.addConstr(Ac[i, :] @ t2 == Config[i])


for i in range(num_edge):
    model.addConstr(Config[node_dict[startIdx[i]]] <= Config[node_dict[endIdx[i]]])







# PCU limits
Par_lane = model.addMVar((Nc), name='Par_lane', vtype=gp.GRB.INTEGER, lb=1)
Par_stage = model.addMVar((Nc), name='Par_stage', vtype=gp.GRB.INTEGER, lb=1)
Par_total = model.addMVar((Nc), name='Par_total', vtype=gp.GRB.INTEGER, lb=1)

for i in range(Nc):
    model.addConstr(Par_lane[i] * Par_stage[i] == Par_total[i])
for i in range(C):
    model.addConstr(Par_total @ Ac[:, i] <= PCU_lim)        






# buffer assignment
for i in range(Nb):
    cin_idx = Nc_dict[Nb_cin[i]]
    cout_idx = Nc_dict[Nb_cout[i]]
    
    for j in range(C):
        t1 = model.addVar(vtype=gp.GRB.BINARY)
        t2 = model.addVar(vtype=gp.GRB.BINARY)
        t3 = model.addVar(vtype=gp.GRB.BINARY)
        t4 = model.addVar(vtype=gp.GRB.BINARY)
        model.addConstr(t1 == gp.and_(Ac[cin_idx, j], Ac[cout_idx, j]))
        model.addConstr(t2 == gp.or_(Ac[cin_idx, j], Ac[cout_idx, j]))
        model.addConstr(t3 == 1 - t1)
        model.addConstr(t4 == gp.and_(t3, t2))
        
        
        model.addConstr((t1 == 1) >> (Ab_onchip[i, j] == 1))
        model.addConstr((t1 == 0) >> (Ab_onchip[i, j] == 0))
        
        
        model.addConstr((t4 == 1) >> (Ab_dram[i, j] == 1))
        model.addConstr((t4 == 0) >> (Ab_dram[i, j] == 0))


for i in range(Nd):
    cout_idx = Nc_dict[Nd_cout[i]]
    
    for j in range(C):
        model.addConstr((Ac[cout_idx, j] == 1) >> (Ad[i, j] == 1))
        model.addConstr((Ac[cout_idx, j] == 0) >> (Ad[i, j] == 0))






# SRAM cap
buffer_depth_original_size = model.addMVar(Nb, name='buffer_depth_original_size', vtype=gp.GRB.INTEGER, lb=1)
buffer_depth_one_size = model.addMVar(Nb, name='buffer_depth_one_size', vtype=gp.GRB.INTEGER, lb=1)
preload_buffer_size = model.addMVar(Nd, name='preload_buffer_size', vtype=gp.GRB.INTEGER, lb=1)

tmpb1 = model.addMVar(Nb, vtype=gp.GRB.INTEGER, lb=1)
tmpb2 = model.addMVar(Nb, vtype=gp.GRB.INTEGER, lb=1)
for i in range(Nb):
    model.addConstr(buffer_depth_original_size[i] >= TSb[i] * D[i])
    model.addConstr(buffer_depth_one_size[i] >= TSb[i] * 2)

tmpd = model.addMVar(Nd, vtype=gp.GRB.INTEGER, lb=1)
for i in range(Nd):
    model.addConstr(preload_buffer_size[i] >= TSd_tiled[i] * 1)

for i in range(C):
    model.addConstr(buffer_depth_original_size @ Ab_onchip[:, i] + buffer_depth_one_size @ Ab_dram[:, i] <= SRAM_Cap)







# compute cycle
Cycle = model.addMVar(Nc, name='Cycle', vtype=gp.GRB.INTEGER, lb=0)
m_factor = model.addMVar(Nc, name='m_factor', vtype=gp.GRB.INTEGER, lb=1)
n_factor = model.addMVar(Nc, name='n_factor', vtype=gp.GRB.INTEGER, lb=1)
    
for i in range(Nc):
    if K[i] == -1:
        model.addConstr(Par_stage[i] == 1)
        model.addConstr(m_factor[i] * Par_lane[i] * VecWidth >= M[i])
        model.addConstr(Cycle[i] == m_factor[i] * N[i])
        
    elif N[i] == 1:
        model.addConstr(Par_stage[i] == 1)
        model.addConstr(m_factor[i] * Par_lane[i] * VecWidth >= M[i])
        model.addConstr(Cycle[i] == m_factor[i] * K[i])
        
    else:
        model.addConstr(m_factor[i] * Par_lane[i] * VecWidth >= M[i])
        model.addConstr(n_factor[i] * Par_stage[i] * StageWidth >= N[i])
        model.addConstr(Cycle[i] == m_factor[i] * n_factor[i] * K[i])       






Compute_Latency = model.addMVar(C, name='Compute_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
DRAM_Latency = model.addMVar(C, name='DRAM_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
Network_Latency = model.addMVar(C, name='Network_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
Setup_Latency = model.addMVar(C, name='Setup_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)

Latency_wo_setup = model.addMVar(C, name='Latency_wo_setup', vtype=gp.GRB.CONTINUOUS, lb=0)
Latency_w_setup = model.addMVar(C, name='Latency_w_setup', vtype=gp.GRB.CONTINUOUS, lb=0)


for i in range(C):
    t1 = model.addMVar(Nc, vtype=gp.GRB.INTEGER, lb=0)
    t2 = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=0)
    
    for j in range(Nc):
        model.addConstr(t1[j] == Cycle[j] * Ac[j, i])
    
    model.addConstr(t2[0] == gp.max_(t1[j] for j in range(Nc)))
            
    model.addConstr(Compute_Latency[i] == t2[0] / Freq * num_tile)
    model.addConstr(DRAM_Latency[i] * DRAM_BW[0] == (TSb @ Ab_dram[:, i]) * num_tile)
    model.addConstr(Network_Latency[i] * Net_BW[0] == (AllReduce @ Ac[:, i]) * allreduce_ratio * num_tile)
    model.addConstr(Latency_wo_setup[i] == gp.max_(Compute_Latency[i], DRAM_Latency[i], Network_Latency[i]))
    
    model.addConstr(Setup_Latency[i] * DRAM_BW[0] == TSd @ Ad[:, i] / batch_size)
    model.addConstr(Latency_w_setup[i] == Latency_wo_setup[i] + Setup_Latency[i])

p2p_latency = model.addMVar(1, name='p2p_latency', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr(p2p_latency[0] * P2P_Net_BW[0] == intermediate)






II = model.addMVar(1, name='II', vtype=gp.GRB.CONTINUOUS, lb=0)
II_normal = model.addMVar(1, name='II_normal', vtype=gp.GRB.CONTINUOUS, lb=0)
II_perfect = model.addMVar(1, name='II_perfect', vtype=gp.GRB.CONTINUOUS, lb=0)



# normal II
tmp = model.addMVar(1, name='tmp', vtype=gp.GRB.CONTINUOUS, lb=0)
# model.addConstr(tmp[0] == (np.ones((C)) @ Latency_w_setup) * num_layer_per_stage)
model.addConstr(tmp[0] == (np.ones((C)) @ Latency_wo_setup) * num_layer_per_stage)
model.addConstr(II_normal[0] == gp.max_(p2p_latency[0], tmp[0]))




# perfect overlap II
total_compute_latency = model.addMVar(1, name='total_compute_latency', vtype=gp.GRB.CONTINUOUS, lb=0)
total_memory_latency = model.addMVar(1, name='total_memory_latency', vtype=gp.GRB.CONTINUOUS, lb=0)
total_network_latency = model.addMVar(1, name='total_network_latency', vtype=gp.GRB.CONTINUOUS, lb=0)
tmp2 = model.addMVar(1, name='tmp2', vtype=gp.GRB.CONTINUOUS, lb=0)
tmp3 = model.addMVar(1, name='tmp3', vtype=gp.GRB.CONTINUOUS, lb=0)

model.addConstr(total_compute_latency[0] == (np.ones((C)) @ Compute_Latency) * num_layer_per_stage)
model.addConstr(total_memory_latency[0] == (np.ones((C)) @ DRAM_Latency) * num_layer_per_stage)
model.addConstr(total_network_latency[0] == (np.ones((C)) @ Network_Latency) * num_layer_per_stage)
model.addConstr(tmp2[0] == gp.max_(total_compute_latency[0], total_memory_latency[0]))
model.addConstr(tmp3[0] == gp.max_(total_network_latency[0], p2p_latency[0]))
model.addConstr(II_perfect[0] == gp.max_(tmp2[0], tmp3[0]))





model.addConstr(II[0] == II_perfect[0])







sample_per_sec = model.addMVar(1, name='sample_per_sec', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr(sample_per_sec[0] * II[0] == 1e9)






# cost
total_dram_cost = model.addMVar((1), name='total_dram_cost', vtype=gp.GRB.CONTINUOUS, lb=0)
total_local_cost = model.addMVar((1), name='total_local_cost', vtype=gp.GRB.CONTINUOUS, lb=0)
total_global_cost = model.addMVar((1), name='total_global_cost', vtype=gp.GRB.CONTINUOUS, lb=0)
cost = model.addMVar((1), name='cost', vtype=gp.GRB.CONTINUOUS, lb=0)

model.addConstr(total_dram_cost[0] == dram_cost*DRAM_BW[0]*64)

model.addConstr(total_local_cost[0] == local_cost*local_link_ar*Net_BW[0] + local_cost*local_link_p2p*P2P_Net_BW[0])

model.addConstr(total_global_cost[0] == global_cost*global_link_ar*Net_BW[0] + global_cost*global_link_p2p*P2P_Net_BW[0])

model.addConstr(cost[0] == total_dram_cost[0] + total_local_cost[0] + total_global_cost[0])



model.setObjective(sample_per_sec[0], gp.GRB.MAXIMIZE)
model.optimize()




for v in model.getVars():
    print(v.varName, v.X)
    
    if v.varName == 'II[0]':
        II = v.X
    if v.varName == 'cost[0]':
        cost = v.X
    if v.varName == 'total_dram_cost[0]':
        total_dram_cost = v.X
    if v.varName == 'total_local_cost[0]':
        total_local_cost = v.X
    if v.varName == 'total_global_cost[0]':
        total_global_cost = v.X    
    if v.varName == 'DRAM_BW[0]':
        DRAM_BW = v.X 
    if v.varName == 'Net_BW[0]':
        Net_BW = v.X  
    if v.varName == 'P2P_Net_BW[0]':
        P2P_Net_BW = v.X 
        
        
FLOP = 0
for i in range(len(M)):
    FLOP += 2*M[i] * K[i] * N[i]
FLOP = FLOP * num_tile * num_layer_per_stage


print(GFLOPS)     
print(FLOP)
print(FLOP/II)
print(FLOP/II/GFLOPS)
print(DRAM_BW)
print(Net_BW)
print(P2P_Net_BW)
print(local_link_ar)
print(local_link_p2p)
print(global_link_ar)
print(global_link_p2p)
print(total_dram_cost)
print(total_local_cost)
print(total_global_cost)
print(cost)
