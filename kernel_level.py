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


word = 2

# read in pd file
dse = setup_pb2.DSE()
with open('./'+name+'/'+'dse_sharded.pb', "rb") as file:
    dse.ParseFromString(file.read())
    
    








# get kernels
class Dim(Enum):
    NO_DIM = 0
    OUTER_DIM = 1
    M_DIM = 2
    K_DIM = 3
    N_DIM = 4

class KernelType(Enum):
    NO_Type = 0
    SYSTOLIC = 1
    SIMD = 2

class CommunicationType(Enum):
    NO_COMMUNICATION = 0
    ALL_REDUCE = 1
    ALL_TO_ALL = 2
    ALL_GATHER = 3

class Optimization(Enum):
    NO_OPTIMIZATION = 0
    FLASHATTENTION = 1
    KERNEL_BY_KERNEL = 2

kernel_name = []
for kernel in dse.dataflow_graph.kernels:
    kernel_name.append(kernel.name)

kernel_type = []
for kernel in dse.dataflow_graph.kernels:
    kernel_type.append(kernel.batch_gemm_elementwise_outer_m_k_n.type)

outer = []
for kernel in dse.dataflow_graph.kernels:
    outer.append(kernel.batch_gemm_elementwise_outer_m_k_n.outer)

M = []
for kernel in dse.dataflow_graph.kernels:
    M.append(kernel.batch_gemm_elementwise_outer_m_k_n.M)

for i in range(len(M)):
    M[i] *= outer[i]

K = []
for kernel in dse.dataflow_graph.kernels:
    K.append(kernel.batch_gemm_elementwise_outer_m_k_n.K)

N = []
for kernel in dse.dataflow_graph.kernels:
    N.append(kernel.batch_gemm_elementwise_outer_m_k_n.N)

sharding = []
for kernel in dse.dataflow_graph.kernels:
    sharding.append(kernel.batch_gemm_elementwise_outer_m_k_n.sharding)


kernel_type = []
for kernel in dse.dataflow_graph.kernels:
    kernel_type.append(kernel.batch_gemm_elementwise_outer_m_k_n.type)

node_communication_type = []
for kernel in dse.dataflow_graph.kernels:
    node_communication_type.append(kernel.batch_gemm_elementwise_outer_m_k_n.communication_type)


node_communication_size = []
for kernel in dse.dataflow_graph.kernels:
    node_communication_size.append(kernel.batch_gemm_elementwise_outer_m_k_n.communication_size)


node_dict = {}
i = 0
for kernel in dse.dataflow_graph.kernels:
    node_dict[kernel.id] = i
    i += 1

num_kernel = len(kernel_name)
seq_len = dse.training.seq_len
num_layer = dse.training.num_layer
Micro_Batch_Size = dse.training.micro_batch_size
opt = dse.training.optimization
Intermediate = M[-1] * N[-1] * word


# get edges
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


edge_communication_type = []
for connection in dse.dataflow_graph.connections:
    edge_communication_type.append(connection.communication_type)


edge_communication_size = []
for connection in dse.dataflow_graph.connections:
    edge_communication_size.append(connection.communication_size)

num_edge = len(startIdx)
    


# get weights
weight_dict = {} # index in weights to node id
weights = []
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size != -1:
        weights.append(kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size)
        weight_dict[i] = kernel.id
        i += 1

num_weight = len(weights)




# get system info
Core = dse.system.accelerator.core
SRAM_Cap = dse.system.accelerator.sram_cap
VecWidth = dse.system.accelerator.systolic_width
StageWidth = dse.system.accelerator.systolic_height
Freq = dse.system.accelerator.freq
dram_bw = dse.system.accelerator.dram_bw
DRAM_Cap = dse.system.accelerator.dram_cap
num_chip = dse.system.num_chip
GFLOPS = 2*VecWidth*StageWidth*Core*Freq


class BasicTopology(Enum):
    NO_BASICTOPOLOGY = 0
    R = 1
    FC = 2
    SW = 3


class Topology(Enum):
    NO_TOPOLOGY = 0
    TORUS_2D = 1
    DRAGONFLY = 2
    DGX_1 = 3
    DGX_2 = 4
    TORUS_3D = 5

if dse.system.topo == Topology.TORUS_2D.value: # 2D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]
    shape = [dse.system.x, dse.system.y]

elif dse.system.topo == Topology.DRAGONFLY.value: # Dragonfly
    topology = [BasicTopology.FC.value, BasicTopology.FC.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]
    shape = [dse.system.x, dse.system.y]

elif dse.system.topo == Topology.DGX_1.value: # DGX-1
    topology = [BasicTopology.FC.value, BasicTopology.SW.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]
    shape = [dse.system.x, dse.system.y]
    
elif dse.system.topo == Topology.DGX_2.value: # DGX-2
    topology = [BasicTopology.SW.value, BasicTopology.SW.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]
    shape = [dse.system.x, dse.system.y]
    
else: # 3D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y, dse.system.link_bw_z]
    shape = [dse.system.x, dse.system.y, dse.system.z]





FLOP = 0.0
for i in range(len(M)):
    if kernel_type[i] == KernelType.SIMD.value:
        FLOP += M[i] * K[i] * N[i]
    else:
        FLOP += 2 * M[i] * K[i] * N[i]
FLOP *= num_layer



C = num_kernel



model = gp.Model()
model.params.NonConvex = 2
model.Params.Threads = 10
model.params.MIPGap = 0.1    # 10%
model.params.TimeLimit = 300  # 5 minutes
model.optimize()






# topology and TP/PP
if len(topology) == 2:
    X = model.addVar(name='X', vtype=gp.GRB.INTEGER)
    Y = model.addVar(name='Y', vtype=gp.GRB.INTEGER)
    model.addConstr(X * Y == num_chip)

    model.addConstr(X == shape[0])
    model.addConstr(Y == shape[1])

    TP = model.addVar(name='TP', vtype=gp.GRB.INTEGER)
    PP = model.addVar(name='PP', vtype=gp.GRB.INTEGER)
    model.addConstr(TP == X)
    model.addConstr(PP == Y)

    Link_BW_X = model.addVar(name='Link_BW_X', vtype=gp.GRB.CONTINUOUS, lb=0)
    Link_BW_Y = model.addVar(name='Link_BW_Y', vtype=gp.GRB.CONTINUOUS, lb=0)
    model.addConstr(Link_BW_X == link_bw[0])
    model.addConstr(Link_BW_Y == link_bw[1])

else:
    X = model.addVar(name='X', vtype=gp.GRB.INTEGER)
    Y = model.addVar(name='Y', vtype=gp.GRB.INTEGER)
    Z = model.addVar(name='Z', vtype=gp.GRB.INTEGER)
    XY = model.addVar(name='XY', vtype=gp.GRB.INTEGER)
    model.addConstr(XY == X * Y)
    model.addConstr(num_chip = XY * Z)

    model.addConstr(X == shape[0])
    model.addConstr(Y == shape[1])
    model.addConstr(Z == shape[2])

    TP = model.addVar(name='TP', vtype=gp.GRB.INTEGER)
    PP = model.addVar(name='PP', vtype=gp.GRB.INTEGER)
    model.addConstr(TP == XY)
    model.addConstr(PP = Z)

    Link_BW_X = model.addVar(name='Link_BW_X', vtype=gp.GRB.CONTINUOUS, lb=0)
    Link_BW_Y = model.addVar(name='Link_BW_Y', vtype=gp.GRB.CONTINUOUS, lb=0)
    Link_BW_Z = model.addVar(name='Link_BW_Z', vtype=gp.GRB.CONTINUOUS, lb=0)
    model.addConstr(Link_BW_X == link_bw[0])
    model.addConstr(Link_BW_Y == link_bw[1])
    model.addConstr(Link_BW_Z == link_bw[2])


# system variables
DRAM_BW = model.addVar(name='DRAM_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr(DRAM_BW == dram_bw)



# pipeline parallelism
layers_per_stage = model.addVar(name='layers_per_stage', vtype=gp.GRB.INTEGER, lb=1)
model.addConstr(layers_per_stage * PP >= num_layer)

aaa = model.addVar(vtype=gp.GRB.BINARY)
intermediate = model.addVar(name='intermediate', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr((aaa == 1) >> (PP == 1))
model.addConstr((aaa == 0) >> (PP >= 2))
model.addConstr((aaa == 1) >> (intermediate == 0))
model.addConstr((aaa == 0) >> (intermediate == Intermediate))


# sharding/tiling kernel
tile_size = model.addVar(name='tile_size', vtype=gp.GRB.INTEGER, lb=1)
num_tile = model.addVar(name='num_tile', vtype=gp.GRB.INTEGER, lb=0)
model.addConstr(tile_size * num_tile == seq_len)

shard_M = model.addMVar(num_kernel, name='shard_M', vtype=gp.GRB.INTEGER, lb=0)
shard_K = model.addMVar(num_kernel, name='shard_K', vtype=gp.GRB.INTEGER, lb=0)
shard_N = model.addMVar(num_kernel, name='shard_N', vtype=gp.GRB.INTEGER, lb=0)

for i in range(num_kernel):
    if sharding[i] == Dim.OUTER_DIM.value or sharding[i] == Dim.M_DIM.value:
        model.addConstr(shard_M[i] * TP >= M[i])
        model.addConstr(shard_K[i] == K[i])
        model.addConstr(shard_N[i] == tile_size)
    elif sharding[i] == Dim.K_DIM.value:
        model.addConstr(shard_M[i] == M[i])
        model.addConstr(shard_K[i] * TP >= K[i])
        model.addConstr(shard_N[i] == tile_size)



# sharding intermediate buffers
shard_intermediate_buffer_size = model.addMVar(num_edge, name='shard_intermediate_buffer_size', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(num_edge):
    upstream_node_idx = node_dict[startIdx[i]]
    model.addConstr(shard_intermediate_buffer_size[i] == shard_M[upstream_node_idx] * shard_N[upstream_node_idx] * word)

# sharding initiation buffers (weights)
shard_initiation_buffer_size = model.addMVar(num_weight, name='shard_initiation_buffer_size', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(num_weight):
    node_idx = node_dict[weight_dict[i]]
    model.addConstr(shard_initiation_buffer_size[i] == shard_M[node_idx] * shard_K[node_idx] * word)



# sharding node communication
shard_node_communication_size = model.addMVar(num_kernel, name='shard_node_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(num_kernel):
    if node_communication_size[i] > 0:
        model.addConstr(shard_node_communication_size[i] * num_tile == node_communication_size[i])
    else:
        model.addConstr(shard_node_communication_size[i] == 0)


# sharding edge communication
shard_edge_communication_size = model.addMVar(num_edge, name='shard_edge_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(num_edge):
    if edge_communication_size[i] > 0:
        model.addConstr(shard_edge_communication_size[i] == shard_intermediate_buffer_size[i])
    else:
        model.addConstr(shard_edge_communication_size[i] == 0)





Config = model.addMVar(num_kernel, name='Config', vtype=gp.GRB.INTEGER, lb=0)
Ab_onchip = model.addMVar((num_edge, C), name='Ab_onchip', vtype=gp.GRB.BINARY) # on-chip
Ab_dram = model.addMVar((num_edge, C), name='Ab_dram', vtype=gp.GRB.BINARY) # to/from DRAM
Ac = model.addMVar((num_kernel, C), name='Ac', vtype=gp.GRB.BINARY)
Ad = model.addMVar((num_weight, C), name='Ad', vtype=gp.GRB.BINARY)



model.addConstr(Config[0] == 0)
model.addConstr(Config[1] == 0)
model.addConstr(Config[2] == 0)
model.addConstr(Config[3] == 1)
model.addConstr(Config[4] == 1)
model.addConstr(Config[5] == 1)
model.addConstr(Config[6] == 1)
model.addConstr(Config[7] == 2)
model.addConstr(Config[8] == 3)

model.addConstr(tile_size == 32)



# kernel assignment   
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


# for kernel-by-kernel
if opt == Optimization.KERNEL_BY_KERNEL.value:
    t3 = np.ones((num_kernel))
    for i in range(C):
        model.addConstr(t3 @ Ac[:, i] >= 1)




# compute resources
Par_lane = model.addMVar((num_kernel), name='Par_lane', vtype=gp.GRB.INTEGER, lb=1)
Par_stage = model.addMVar((num_kernel), name='Par_stage', vtype=gp.GRB.INTEGER, lb=1)
Par_total = model.addMVar((num_kernel), name='Par_total', vtype=gp.GRB.INTEGER, lb=1)

for i in range(num_kernel):
    model.addConstr(Par_lane[i] * Par_stage[i] == Par_total[i])
for i in range(C):
    model.addConstr(Par_total @ Ac[:, i] <= Core)        






# intermediate buffer assignment
for i in range(num_edge):
    start_node_idx = node_dict[startIdx[i]]
    end_node_idx = node_dict[endIdx[i]]
    
    for j in range(C):
        t1 = model.addVar(vtype=gp.GRB.BINARY)
        t2 = model.addVar(vtype=gp.GRB.BINARY)
        t3 = model.addVar(vtype=gp.GRB.BINARY)
        t4 = model.addVar(vtype=gp.GRB.BINARY)

        model.addConstr(t1 == gp.and_(Ac[start_node_idx, j], Ac[end_node_idx, j]))
        model.addConstr(t2 == gp.or_(Ac[start_node_idx, j], Ac[end_node_idx, j]))
        model.addConstr(t3 == 1 - t1)
        model.addConstr(t4 == gp.and_(t3, t2))
        
        model.addConstr((t1 == 1) >> (Ab_onchip[i, j] == 1))
        model.addConstr((t1 == 0) >> (Ab_onchip[i, j] == 0))
        
        model.addConstr((t4 == 1) >> (Ab_dram[i, j] == 1))
        model.addConstr((t4 == 0) >> (Ab_dram[i, j] == 0))


# initiation buffer assignment
for i in range(num_weight):
    node_idx = node_dict[weight_dict[i]]
    
    for j in range(C):
        model.addConstr((Ac[node_idx, j] == 1) >> (Ad[i, j] == 1))
        model.addConstr((Ac[node_idx, j] == 0) >> (Ad[i, j] == 0))






# SRAM cap
shard_intermediate_buffer_size_depth_original = model.addMVar(num_edge, name='shard_intermediate_buffer_size_depth_original', vtype=gp.GRB.INTEGER, lb=0)
shard_intermediate_buffer_size_depth_two = model.addMVar(num_edge, name='shard_intermediate_buffer_size_depth_two', vtype=gp.GRB.INTEGER, lb=0)
shard_initiation_buffer_size_depth_one = model.addMVar(num_weight, name='shard_initiation_buffer_size_depth_one', vtype=gp.GRB.INTEGER, lb=0)


for i in range(num_edge):
    model.addConstr(shard_intermediate_buffer_size_depth_original[i] >= shard_intermediate_buffer_size[i] * depth[i])
    model.addConstr(shard_intermediate_buffer_size_depth_two[i] >= shard_intermediate_buffer_size[i] * 2)

for i in range(num_weight):
    model.addConstr(shard_initiation_buffer_size_depth_one[i] >= shard_initiation_buffer_size[i] * 1)

for i in range(C):
    model.addConstr(shard_intermediate_buffer_size_depth_original @ Ab_onchip[:, i] + shard_intermediate_buffer_size_depth_two @ Ab_dram[:, i] + shard_initiation_buffer_size_depth_one @ Ad[:, i] <= SRAM_Cap)




# dram cap
Micro_Batch_Size = model.addVar(name='Micro_Batch_Size', vtype=gp.GRB.INTEGER, lb=1)
dram_bytes_per_config_intermediate = model.addMVar(C, name='dram_bytes_per_config_intermediate', vtype=gp.GRB.CONTINUOUS, lb=0)
dram_bytes_per_config_initiation = model.addMVar(C, name='dram_bytes_per_config_initiation', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    model.addConstr(dram_bytes_per_config_intermediate[i] == shard_intermediate_buffer_size @ Ab_dram[:, i])
    model.addConstr(dram_bytes_per_config_initiation[i] == shard_initiation_buffer_size @ Ad[:, i])

dram_bytes_per_batch_1 = model.addVar(name='dram_bytes_per_batch_1', vtype=gp.GRB.CONTINUOUS, lb=1)
model.addConstr(dram_bytes_per_batch_1 == (np.ones((C)) @ dram_bytes_per_config_intermediate + np.ones((C)) @ dram_bytes_per_config_initiation) * layers_per_stage)
model.addConstr(dram_bytes_per_batch_1 * Micro_Batch_Size <= DRAM_Cap)



# compute cycle
Cycle = model.addMVar(num_kernel, name='Cycle', vtype=gp.GRB.INTEGER, lb=0)
m_factor = model.addMVar(num_kernel, name='m_factor', vtype=gp.GRB.INTEGER, lb=1)
n_factor = model.addMVar(num_kernel, name='n_factor', vtype=gp.GRB.INTEGER, lb=1)
    
for i in range(num_kernel):
    if kernel_type[i] == KernelType.SIMD.value:
        model.addConstr(m_factor[i] * Par_lane[i] * VecWidth >= shard_M[i])
        model.addConstr(Par_stage[i] == 1)
        model.addConstr(Cycle[i] == m_factor[i] * shard_N[i])
    else:
        model.addConstr(m_factor[i] * Par_lane[i] * VecWidth >= shard_M[i])
        model.addConstr(n_factor[i] * Par_stage[i] * StageWidth >= shard_N[i])

        tmp = model.addVar(vtype=gp.GRB.INTEGER)
        model.addConstr(tmp == m_factor[i] * n_factor[i])
        model.addConstr(Cycle[i] == tmp * shard_K[i])
            






Compute_Latency = model.addMVar(num_kernel, name='Compute_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    t1 = model.addMVar(num_kernel, vtype=gp.GRB.INTEGER, lb=0)
    t2 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    for j in range(num_kernel):
        model.addConstr(t1[j] == Cycle[j] * Ac[j, i])
    model.addConstr(t2 == gp.max_(t1[j] for j in range(num_kernel)))
    model.addConstr(Compute_Latency[i] == t2 * num_tile / Freq)

DRAM_Latency = model.addMVar(num_kernel, name='DRAM_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    t1 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(t1 == shard_intermediate_buffer_size @ Ab_dram[:, i])
    model.addConstr(DRAM_Latency[i] * DRAM_BW == t1 * num_tile)

allreduce_ratio = 1
Network_Latency = model.addMVar(num_kernel, name='Network_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    t1 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(t1 == Ac[:, i] @ shard_node_communication_size)
    model.addConstr(Network_Latency[i] * Link_BW_X == t1 * allreduce_ratio * num_tile)

Latency_wo_setup = model.addMVar(C, name='Latency_wo_setup', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    model.addConstr(Latency_wo_setup[i] == gp.max_(Compute_Latency[i], DRAM_Latency[i], Network_Latency[i]))

Setup_Latency = model.addMVar(num_kernel, name='Setup_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(aaa == DRAM_BW * Micro_Batch_Size)
    model.addConstr(Setup_Latency[i] * aaa == shard_initiation_buffer_size @ Ad[:, i])

Per_Config_II = model.addMVar(C, name='Per_Config_II', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    model.addConstr(Per_Config_II[i] == Latency_wo_setup[i] + Setup_Latency[i])

p2p_latency = model.addVar(name='p2p_latency', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr(p2p_latency * Link_BW_Y == intermediate)



II = model.addVar(name='II', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr(II == np.ones((C)) @ Per_Config_II * layers_per_stage + p2p_latency)


util = model.addVar(name='util', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr(util * II * GFLOPS * num_chip == FLOP)


# sample_per_sec = model.addMVar(1, name='sample_per_sec', vtype=gp.GRB.CONTINUOUS, lb=0)
# model.addConstr(sample_per_sec[0] * II[0] == 1e9 * micro_batch_size)


model.setObjective(util, gp.GRB.MAXIMIZE)
model.optimize()


# get values from gurobi
shard_M = []
shard_K = []
shard_N = []
shard_intermediate_buffer_size = []
shard_initiation_buffer_size = []
for v in model.getVars():
    print(v.varName, v.X)
    
    if v.varName.startswith('shard_M'):
        shard_M.append(v.X)
    if v.varName.startswith('shard_K'):
        shard_K.append(v.X)
    if v.varName.startswith('shard_N'):
        shard_N.append(v.X)
    if v.varName.startswith('shard_intermediate_buffer_size'):
        shard_intermediate_buffer_size.append(v.X)
    if v.varName.startswith('shard_initiation_buffer_size'):
        shard_initiation_buffer_size.append(v.X)
    if v.varName.startswith('II'):
        II = v.X


print('GFLOPS', GFLOPS)
print('FLOP', FLOP)



# update kernels
i = 0
for kernel in dse.dataflow_graph.kernels:
    kernel.batch_gemm_elementwise_outer_m_k_n.shard_outer_M = int(shard_M[i])
    kernel.batch_gemm_elementwise_outer_m_k_n.shard_K = int(shard_K[i])
    kernel.batch_gemm_elementwise_outer_m_k_n.shard_N = int(shard_N[i])
    i += 1


# update edges
i = 0
for connection in dse.dataflow_graph.connections:
    connection.shard_tensor_size = float(shard_intermediate_buffer_size[i])
    i += 1


# update weights
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size != -1:
        kernel.batch_gemm_elementwise_outer_m_k_n.shard_weight_size = float(shard_initiation_buffer_size[i])
        i += 1





# write to final binary
with open('./'+name+'/'+'dse_final.pb', "wb") as file:
    file.write(dse.SerializeToString())


# write to final text file
with open('./'+name+'/'+'dse_final.txt', "w") as file:
    text_format.PrintMessage(dse, file)


# create dot graph
node_list = []
edge_list = []
dict = {}
graph = pydot.Dot(graph_type='digraph')
for kernel in dse.dataflow_graph.kernels:  
    label = text_format.MessageToString(kernel)
    pydot_node = pydot.Node(kernel.name, style="filled", fillcolor="red", label=label)
    dict[kernel.id] = pydot_node
    graph.add_node(pydot_node)

for connection in dse.dataflow_graph.connections:
    label = text_format.MessageToString(connection)
    pydot_edge = pydot.Edge(dict[connection.startIdx], dict[connection.endIdx], label=label)
    graph.add_edge(pydot_edge)


graph.write_png('./'+name+'/'+'dataflow_graph_final.png') 

