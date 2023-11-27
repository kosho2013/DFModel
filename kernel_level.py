import gurobipy as gp
import argparse
import numpy as np
import setup_pb2
import pprint
from enum import Enum
from google.protobuf import text_format
import pydot
import copy
import sys


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
    NO_SHARDING = 5

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


class FWD_BWD(Enum):
    Placeholder = 0
    FWD = 1
    BWD = 2




kernel_name = []   
kernel_type = []
outer = []
M = []
K = []
N = []
sharding = []
configs = []
node_communication_type = []
node_communication_size = []
fwd_bwd = []
topological_number = []
weight_tensor_size = []
for kernel in dse.dataflow_graph.kernels:
    if kernel.WhichOneof('kernel_variant') == 'batch_gemm_elementwise_outer_m_k_n':
        kernel_type.append(kernel.batch_gemm_elementwise_outer_m_k_n.type)
        outer.append(kernel.batch_gemm_elementwise_outer_m_k_n.outer)
        M.append(kernel.batch_gemm_elementwise_outer_m_k_n.M * kernel.batch_gemm_elementwise_outer_m_k_n.outer)
        K.append(kernel.batch_gemm_elementwise_outer_m_k_n.K)
        N.append(kernel.batch_gemm_elementwise_outer_m_k_n.N)
        sharding.append(kernel.batch_gemm_elementwise_outer_m_k_n.sharding)
        node_communication_type.append(kernel.batch_gemm_elementwise_outer_m_k_n.communication_type)
        node_communication_size.append(kernel.batch_gemm_elementwise_outer_m_k_n.communication_size)
        fwd_bwd.append(kernel.batch_gemm_elementwise_outer_m_k_n.fwd_bwd)
        weight_tensor_size.append(kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size)
    configs.append(kernel.config)
    kernel_name.append(kernel.name)
    topological_number.append(kernel.topological_number)



node_dict = {}
i = 0
for kernel in dse.dataflow_graph.kernels:
    node_dict[kernel.id] = i
    i += 1

num_kernel = len(kernel_name)
hidden_dim = dse.training.hidden_dim
head_dim = dse.training.head_dim
num_head = dse.training.num_head
seq_len = dse.training.seq_len
num_layer = dse.training.num_layer
num_config = dse.training.num_config
seq_tile_size = dse.training.seq_tile_size
opt = dse.training.optimization
Intermediate = hidden_dim * seq_len * word





last_fwd_kernel = -1
first_bwd_kernel = -1
for i in range(num_kernel):
    if fwd_bwd[i] == FWD_BWD.FWD.value:
        last_fwd_kernel = i

for i in range(num_kernel):       
    if fwd_bwd[i] == FWD_BWD.BWD.value:
        first_bwd_kernel = i
        break





if num_head * head_dim != hidden_dim:
    raise Exception('Wrong!')



# get edges
startIdx = []
endIdx = []
depth = []
tensor_size = []
for connection in dse.dataflow_graph.connections:
    startIdx.append(connection.startIdx)
    endIdx.append(connection.endIdx)
    depth.append(connection.buffer_depth)
    tensor_size.append(connection.tensor_size)

num_edge = len(startIdx)
    


# get weights
weight_dict = {} # index in weights to node id
weights = []
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size != -1:
        if kernel.WhichOneof('kernel_variant') == 'batch_gemm_elementwise_outer_m_k_n':
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
    DRAGONFLY_2D = 2
    DGX_1_2D = 3
    DGX_2_2D = 4
    TORUS_3D = 5
    SINGLE_CHIP = 6
    FC_1D = 7
    RING_1D = 8
    


    
if dse.system.topo == Topology.SINGLE_CHIP.value: # single chip
    topology = []
    link_bw = []
    shape = []

elif dse.system.topo == Topology.FC_1D.value: # 1D FC
    topology = [BasicTopology.FC.value]
    link_bw = [dse.system.link_bw_x]
    shape = [dse.system.x]
  
elif dse.system.topo == Topology.RING_1D.value: # 1D Ring
    topology = [BasicTopology.R.value]
    link_bw = [dse.system.link_bw_x]
    shape = [dse.system.x]    
    
elif dse.system.topo == Topology.TORUS_2D.value: # 2D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]
    shape = [dse.system.x, dse.system.y]

elif dse.system.topo == Topology.DRAGONFLY_2D.value: # 2D Dragonfly
    topology = [BasicTopology.FC.value, BasicTopology.FC.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]
    shape = [dse.system.x, dse.system.y]

elif dse.system.topo == Topology.DGX_1_2D.value: # 2D DGX-1
    topology = [BasicTopology.FC.value, BasicTopology.SW.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]
    shape = [dse.system.x, dse.system.y]
    
elif dse.system.topo == Topology.DGX_2_2D.value: # 2D DGX-2
    topology = [BasicTopology.SW.value, BasicTopology.SW.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y]
    shape = [dse.system.x, dse.system.y]
    
elif dse.system.topo == Topology.TORUS_3D.value: # 3D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.link_bw_x, dse.system.link_bw_y, dse.system.link_bw_z]
    shape = [dse.system.x, dse.system.y, dse.system.z]

else:
    raise Exception('Wrong!')









model = gp.Model()
model.params.NonConvex = 2
model.Params.Threads = 128
model.params.MIPGap = 1e-10
# model.params.TimeLimit = 80






# topology and TP/PP
TP = model.addVar(name='TP', vtype=gp.GRB.INTEGER)
PP = model.addVar(name='PP', vtype=gp.GRB.INTEGER)
    
if len(topology) == 0: # single chip
    X = model.addVar(name='X', vtype=gp.GRB.INTEGER)
    Y = model.addVar(name='Y', vtype=gp.GRB.INTEGER)
    model.addConstr(X == 1)
    model.addConstr(Y == 1)

    model.addConstr(TP == X)
    model.addConstr(PP == Y)    
    
    Link_BW_X = model.addVar(name='Link_BW_X', vtype=gp.GRB.CONTINUOUS, lb=0)
    Link_BW_Y = model.addVar(name='Link_BW_Y', vtype=gp.GRB.CONTINUOUS, lb=0)
    model.addConstr(Link_BW_X == sys.maxsize)
    model.addConstr(Link_BW_Y == sys.maxsize)
    
elif len(topology) == 1: # 1D
    X = model.addVar(name='X', vtype=gp.GRB.INTEGER)
    Y = model.addVar(name='Y', vtype=gp.GRB.INTEGER)
    
    model.addConstr(X == num_chip)
    model.addConstr(Y == 1)

    model.addConstr(TP == X)
    model.addConstr(PP == Y)
    
    Link_BW_X = model.addVar(name='Link_BW_X', vtype=gp.GRB.CONTINUOUS, lb=0)
    Link_BW_Y = model.addVar(name='Link_BW_Y', vtype=gp.GRB.CONTINUOUS, lb=0)
    if link_bw[0] == 0: # DSE
        pass
    else:
        model.addConstr(Link_BW_X == link_bw[0])
    
    model.addConstr(Link_BW_Y == sys.maxsize)
        
elif len(topology) == 2: # 2D
    X = model.addVar(name='X', vtype=gp.GRB.INTEGER)
    Y = model.addVar(name='Y', vtype=gp.GRB.INTEGER)
    
    if shape[0] == 0 and shape[1] == 0: # DSE on topology dimensions
        pass
    else:
        model.addConstr(X == shape[0])
        model.addConstr(Y == shape[1])
    model.addConstr(X * Y == num_chip)

    model.addConstr(TP == X)
    model.addConstr(PP == Y)
    
    Link_BW_X = model.addVar(name='Link_BW_X', vtype=gp.GRB.CONTINUOUS, lb=0)
    Link_BW_Y = model.addVar(name='Link_BW_Y', vtype=gp.GRB.CONTINUOUS, lb=0)
    if link_bw[0] == 0 and link_bw[1] == 0: # DSE
        pass
    else:
        model.addConstr(Link_BW_X == link_bw[0])
        model.addConstr(Link_BW_Y == link_bw[1])

elif len(topology) == 3: # 3D
    X = model.addVar(name='X', vtype=gp.GRB.INTEGER)
    Y = model.addVar(name='Y', vtype=gp.GRB.INTEGER)
    Z = model.addVar(name='Z', vtype=gp.GRB.INTEGER)
    XY = model.addVar(name='XY', vtype=gp.GRB.INTEGER)
    model.addConstr(XY == X * Y)
    model.addConstr(num_chip = XY * Z)
    
    if shape[0] == 0 and shape[1] == 0 and shape[2] == 0: # DSE on topology dimensions
        pass
    else:
        model.addConstr(X == shape[0])
        model.addConstr(Y == shape[1])
        model.addConstr(Z == shape[2])

    model.addConstr(TP == XY)
    model.addConstr(PP = Z)

    Link_BW_X = model.addVar(name='Link_BW_X', vtype=gp.GRB.CONTINUOUS, lb=0)
    Link_BW_Y = model.addVar(name='Link_BW_Y', vtype=gp.GRB.CONTINUOUS, lb=0)
    Link_BW_Z = model.addVar(name='Link_BW_Z', vtype=gp.GRB.CONTINUOUS, lb=0)
    if link_bw[0] == 0 and link_bw[1] == 0 and link_bw[2] == 0: # DSE
        pass
    else:
        model.addConstr(Link_BW_X == link_bw[0])
        model.addConstr(Link_BW_Y == link_bw[1])
        model.addConstr(Link_BW_Z == link_bw[2])

else:
    raise Exception('Wrong!')





DRAM_BW = model.addVar(name='DRAM_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
if dram_bw == 0: # DSE
    pass
else:
    model.addConstr(DRAM_BW == dram_bw)











# sharding/tiling kernel
weight_tiling = model.addMVar(num_kernel, name='weight_tiling', vtype=gp.GRB.INTEGER, lb=1)

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
    elif sharding[i] == Dim.K_DIM.value:
        model.addConstr(shard_M[i] == M[i])
        model.addConstr(shard_K[i] * TP >= K[i])
    elif sharding[i] == Dim.NO_SHARDING.value:
        model.addConstr(shard_M[i] == M[i])
        model.addConstr(shard_K[i] == K[i])
    else:
        raise Exception('Wrong!')
        
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
        model.addConstr(shard_node_communication_size[i] == shard_M[i] * shard_N[i] * word)
    else:
        model.addConstr(shard_node_communication_size[i] == 0)





if num_config == 0: # not specified
    C = num_kernel
else:
    C = num_config
    
    
    
    
Config = model.addMVar(num_kernel, name='Config', vtype=gp.GRB.INTEGER, lb=0)
Ab_onchip = model.addMVar((num_edge, C), name='Ab_onchip', vtype=gp.GRB.BINARY) # on-chip
Ab_dram = model.addMVar((num_edge, C), name='Ab_dram', vtype=gp.GRB.BINARY) # to/from DRAM
Ac = model.addMVar((num_kernel, C), name='Ac', vtype=gp.GRB.BINARY)
Ad = model.addMVar((num_weight, C), name='Ad', vtype=gp.GRB.BINARY)




for i in range(len(configs)):
    if configs[i] == -1: # not specified
        pass
    else:
        model.addConstr(Config[i] == configs[i])


if seq_tile_size == 0: # DSE
    pass
else:
    model.addConstr(tile_size == dse.training.seq_tile_size)


# kernel assignment   
for i in range(num_kernel):
    model.addConstr(Ac[i, :] @ np.ones((C)) == 1)
    
    
t2 = np.zeros((C))
for i in range(C):
    t2[i] = i
for i in range(num_kernel):
    model.addConstr(Ac[i, :] @ t2 == Config[i])


for i in range(num_edge):
    model.addConstr(Config[node_dict[startIdx[i]]] <= Config[node_dict[endIdx[i]]])




# fwd and bwd not in the same config
if first_bwd_kernel == -1:
    pass
else:
    model.addConstr(Config[first_bwd_kernel] - Config[last_fwd_kernel] >= 1)


if opt == Optimization.KERNEL_BY_KERNEL.value: # kernel-by-kernel
    for i in range(C):
        model.addConstr(np.ones((num_kernel)) @ Ac[:, i] >= 1)
    model.addConstr(num_config == num_kernel)
        
elif opt == Optimization.FLASHATTENTION.value:
    for i in range(num_kernel):
        model.addConstr(weight_tiling[i] == 1)

else:
    raise Exception('Wrong!')


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
    node_idx = node_dict[weight_dict[i]]
    
    model.addConstr(shard_initiation_buffer_size_depth_one[i] * weight_tiling[node_idx] >= shard_initiation_buffer_size[i] * 1)

for i in range(C):
    model.addConstr(shard_intermediate_buffer_size_depth_original @ Ab_onchip[:, i] + shard_intermediate_buffer_size_depth_two @ Ab_dram[:, i] + shard_initiation_buffer_size_depth_one @ Ad[:, i] <= SRAM_Cap)




# dram cap
Micro_Batch_Size = model.addVar(name='Micro_Batch_Size', vtype=gp.GRB.INTEGER, lb=1)
dram_bytes_per_config_intermediate = model.addMVar(C, name='dram_bytes_per_config_intermediate', vtype=gp.GRB.CONTINUOUS, lb=0)
dram_bytes_per_config_initiation = model.addMVar(C, name='dram_bytes_per_config_initiation', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(aaa == (shard_intermediate_buffer_size @ Ab_dram[:, i]))
    model.addConstr(dram_bytes_per_config_intermediate[i] == aaa * num_tile)
    model.addConstr(dram_bytes_per_config_initiation[i] == shard_initiation_buffer_size @ Ad[:, i])

dram_bytes_initiation = model.addVar(name='dram_bytes_initiation', vtype=gp.GRB.CONTINUOUS, lb=0)
dram_bytes_intermediate = model.addVar(name='dram_bytes_intermediate', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr(dram_bytes_initiation == np.ones((C)) @ dram_bytes_per_config_initiation)
model.addConstr(dram_bytes_intermediate == np.ones((C)) @ dram_bytes_per_config_intermediate)
model.addConstr((dram_bytes_initiation + dram_bytes_intermediate * Micro_Batch_Size) * num_layer <= DRAM_Cap * PP)





# record the weight tiling factor for kernel-by-kernel
if opt == Optimization.KERNEL_BY_KERNEL.value:
    weight_tiling_per_config = model.addMVar(C, name='weight_tiling_per_config', vtype=gp.GRB.INTEGER, lb=1)
    for i in range(C):
        model.addConstr(weight_tiling_per_config[i] == Ac[:, i] @ weight_tiling)
        



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


DRAM_bytes = model.addMVar(C, name='DRAM_bytes', vtype=gp.GRB.CONTINUOUS, lb=0)
DRAM_Latency = model.addMVar(num_kernel, name='DRAM_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    t1 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(t1 == shard_intermediate_buffer_size @ Ab_dram[:, i])
    
    if opt == Optimization.KERNEL_BY_KERNEL.value:
        aaa = model.addVar(vtype=gp.GRB.INTEGER)
        model.addConstr(aaa == num_tile * weight_tiling_per_config[i])
        model.addConstr(DRAM_Latency[i] * DRAM_BW == t1 * aaa)
        model.addConstr(DRAM_bytes[i] == t1 * aaa)
    
    elif opt == Optimization.FLASHATTENTION.value:
        model.addConstr(DRAM_Latency[i] * DRAM_BW == t1 * num_tile)
        model.addConstr(DRAM_bytes[i] == t1 * num_tile)
        
    else:
        raise Exception('Wrong!')
    
    
total_DRAM_bytes = model.addVar(name='total_DRAM_bytes', vtype=gp.GRB.CONTINUOUS)  
model.addConstr(total_DRAM_bytes == np.ones((C)) @ DRAM_bytes)



allreduce_ratio = model.addVar(name='allreduce_ratio', vtype=gp.GRB.CONTINUOUS) 
if len(topology) == 0:
    model.addConstr(allreduce_ratio == 0)

elif len(topology) == 1 or len(topology) == 2:
    if topology[0] == BasicTopology.R.value:
        model.addConstr(allreduce_ratio * TP == TP - 1)
    elif topology[0] == BasicTopology.FC.value:
        model.addConstr(allreduce_ratio * TP == 2)
    elif topology[0] == BasicTopology.SW.value:
        model.addConstr(allreduce_ratio * TP == TP - 1)
    else:
        raise Exception('Wrong!')

elif len(topology) == 3:
    pass

else:
    raise Exception('Wrong!')



# communication created by tensor parallelism 
Network_bytes = model.addMVar(C, name='Network_bytes', vtype=gp.GRB.CONTINUOUS, lb=0)
Network_Latency = model.addMVar(num_kernel, name='Network_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    t1 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(t1 == Ac[:, i] @ shard_node_communication_size)
    t2 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(t2 == allreduce_ratio * num_tile)
    model.addConstr(Network_Latency[i] * Link_BW_X == t1 * t2)
    model.addConstr(Network_bytes[i] == t1 * t2)

total_Network_bytes = model.addVar(name='total_Network_bytes', vtype=gp.GRB.CONTINUOUS) 
model.addConstr(total_Network_bytes == np.ones((C)) @ Network_bytes)


# communication created by pipeline parallelism
aaa = model.addVar(vtype=gp.GRB.BINARY)
intermediate = model.addVar(name='intermediate', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr((aaa == 1) >> (PP == 1))
model.addConstr((aaa == 0) >> (PP >= 2))
model.addConstr((aaa == 1) >> (intermediate == 0))
model.addConstr((aaa == 0) >> (intermediate == Intermediate))

p2p_latency = model.addVar(name='p2p_latency', vtype=gp.GRB.CONTINUOUS)
model.addConstr(p2p_latency * Link_BW_Y == intermediate)


# weight loading overheads
Setup_Latency = model.addMVar(num_kernel, name='Setup_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(aaa == DRAM_BW * Micro_Batch_Size)
    model.addConstr(Setup_Latency[i] * aaa == shard_initiation_buffer_size @ Ad[:, i])




Latency_wo_setup = model.addMVar(C, name='Latency_wo_setup', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    model.addConstr(Latency_wo_setup[i] == gp.max_(Compute_Latency[i], DRAM_Latency[i], Network_Latency[i]))

Per_Config_II = model.addMVar(C, name='Per_Config_II', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    model.addConstr(Per_Config_II[i] == Latency_wo_setup[i] + Setup_Latency[i])
    




aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
II = model.addVar(name='II', vtype=gp.GRB.CONTINUOUS)
model.addConstr(aaa == np.ones((C)) @ Per_Config_II)
model.addConstr(II == gp.max_(aaa, p2p_latency))



model.setObjective(II, gp.GRB.MINIMIZE)
model.optimize()



# get values from gurobi
shard_M = []
shard_K = []
shard_N = []
shard_intermediate_buffer_size = []
shard_initiation_buffer_size = []
Config = []
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
    if v.varName.startswith('total_DRAM_bytes'):
        total_DRAM_bytes = v.X
    if v.varName.startswith('total_Network_bytes'):
        total_Network_bytes = v.X
    if v.varName.startswith('TP'):
        TP = v.X
    if v.varName.startswith('PP'):
        PP = v.X
    if v.varName.startswith('Config'):
        Config.append(v.X)
        



FLOP = 0.0
for i in range(len(M)):
    if kernel_type[i] == KernelType.SIMD.value:
        FLOP += M[i] * K[i] * N[i]
    else:
        FLOP += 2 * M[i] * K[i] * N[i]
FLOP *= num_layer
        
layers_per_stage = num_layer / PP
II *= layers_per_stage

   
print('GFLOPS', GFLOPS)
print('II', II)
print('Samples/s', 1e9/II)
print('util', FLOP/II/GFLOPS/num_chip)
if total_DRAM_bytes == 0:
    print('oim infinity')
else:
    print('OI Memory', FLOP / num_chip / total_DRAM_bytes / layers_per_stage)
if total_Network_bytes == 0:
    print('oin infinity')
else:
    print('OI network', FLOP / num_chip / total_Network_bytes / layers_per_stage)




# update kernels
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.WhichOneof('kernel_variant') == 'batch_gemm_elementwise_outer_m_k_n':
        kernel.batch_gemm_elementwise_outer_m_k_n.shard_outer_M = int(shard_M[i])
        kernel.batch_gemm_elementwise_outer_m_k_n.shard_K = int(shard_K[i])
        kernel.batch_gemm_elementwise_outer_m_k_n.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
        i += 1


# update edges
i = 0
for connection in dse.dataflow_graph.connections:
    connection.shard_tensor_size = float(shard_intermediate_buffer_size[i])
    i += 1


# update weights
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.WhichOneof('kernel_variant') == 'batch_gemm_elementwise_outer_m_k_n':
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

