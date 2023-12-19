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



# read in pd file
dse = setup_pb2.DSE()
with open('./'+name+'/'+'dse_sharded.pb', "rb") as file:
    dse.ParseFromString(file.read())
    
    


# get kernels
class Dim(Enum):
    DIM_PLACEHOLDER = 0
    OUTER_DIM = 1
    M_DIM = 2
    K_DIM = 3
    N_DIM = 4
    NO_DIM = 5

class KernelType(Enum):
    NO_Type = 0
    SYSTOLIC = 1
    SIMD = 2

class Communication(Enum):
    NO_COMMUNICATION = 0
    ALL_REDUCE = 1
    ALL_TO_ALL = 2
    ALL_GATHER = 3
    ALL_REDUCE_PERIODIC = 4
    POINT_TO_POINT = 5
    BROADCAST = 6
    
class Optimization(Enum):
    NO_OPTIMIZATION = 0
    FLASHATTENTION = 1
    KERNEL_BY_KERNEL = 2


class FWD_BWD(Enum):
    Placeholder = 0
    FWD = 1
    BWD = 2

class Objective(Enum):
    Objective_Placeholder = 0
    PERFORMANCE = 1
    COST = 2
    
class BasicTopology(Enum):
    NO_BASICTOPOLOGY = 0
    R = 1
    FC = 2
    SW = 3
    
class Sync_Async(Enum):
    NO_TRAINING = 0;
    SYNC = 1;
    ASYNC = 2;
    
    
kernel_id = []
kernel_name = []   
kernel_type = []
configs = []
fwd_bwd = []
topological_number = []

M = []
K = []
N = []
weight_tensor_size = []

sharding = []
node_communication_type = []
node_communication_size = []
memory_size = []

tiling = []
node_dict = {}
i = 0
for kernel in dse.dataflow_graph.kernels:
    kernel_id.append(kernel.id)
    kernel_name.append(kernel.name)
    kernel_type.append(kernel.type)
    fwd_bwd.append(kernel.fwd_bwd)
    configs.append(kernel.config)
    topological_number.append(kernel.topological_number)
    
    if kernel.WhichOneof('kernel_variant') == 'gemm_input1_weight':  
        M.append(kernel.gemm_input1_weight.outer * kernel.gemm_input1_weight.M)
        K.append(kernel.gemm_input1_weight.K)
        N.append(kernel.gemm_input1_weight.N)
        
        weight_tensor_size.append(kernel.gemm_input1_weight.weight_tensor_size)
        
        sharding.append(kernel.gemm_input1_weight.sharding)
        node_communication_type.append(kernel.gemm_input1_weight.communication_type)
        node_communication_size.append(kernel.gemm_input1_weight.communication_size)
        
        tiling.append(kernel.gemm_input1_weight.tiling)
        
        memory_size.append(kernel.gemm_input1_weight.memory_size)
        
    elif kernel.WhichOneof('kernel_variant') == 'gemm_input1_input2':
        M.append(kernel.gemm_input1_input2.outer * kernel.gemm_input1_input2.M)
        K.append(kernel.gemm_input1_input2.K)
        N.append(kernel.gemm_input1_input2.N)
        
        weight_tensor_size.append(-1.0)
        
        sharding.append(kernel.gemm_input1_input2.sharding)
        node_communication_type.append(kernel.gemm_input1_input2.communication_type)
        node_communication_size.append(kernel.gemm_input1_input2.communication_size)
        
        tiling.append(kernel.gemm_input1_input2.tiling)
        
        memory_size.append(kernel.gemm_input1_input2.memory_size)

    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1':
        M.append(kernel.elementwise_input1.outer * kernel.elementwise_input1.M)
        K.append(1)
        N.append(kernel.elementwise_input1.N)
        
        weight_tensor_size.append(-1.0)
        
        sharding.append(kernel.elementwise_input1.sharding)
        node_communication_type.append(kernel.elementwise_input1.communication_type)
        node_communication_size.append(kernel.elementwise_input1.communication_size)
        
        tiling.append(kernel.elementwise_input1.tiling)
        
        memory_size.append(kernel.elementwise_input1.memory_size)
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1_input2':
        M.append(kernel.elementwise_input1_input2.outer * kernel.elementwise_input1_input2.M)
        K.append(1)
        N.append(kernel.elementwise_input1_input2.N)
        
        weight_tensor_size.append(-1.0)
        
        sharding.append(kernel.elementwise_input1_input2.sharding)
        node_communication_type.append(kernel.elementwise_input1_input2.communication_type)
        node_communication_size.append(kernel.elementwise_input1_input2.communication_size)
        
        tiling.append(kernel.elementwise_input1_input2.tiling)
        
        memory_size.append(kernel.elementwise_input1_input2.memory_size)
            
    else:
        raise Exception('Wrong!')
    
    node_dict[kernel.id] = i
    i += 1


memory_size = np.array(memory_size)


# get weights
weight_dict = {} # index in weights to node id
cnt = 0
for i in range(len(weight_tensor_size)):
    if weight_tensor_size[i] != -1:
        weight_dict[cnt] = kernel_id[i]
        cnt += 1
num_weight = len(weight_dict.keys())










num_kernel = len(kernel_name)


if dse.training.WhichOneof('workload_variant') == 'llm':
    hidden_dim = dse.training.llm.hidden_dim
    head_dim = dse.training.llm.head_dim
    num_head = dse.training.llm.num_head
    seq_len = dse.training.llm.seq_len
    num_layer = dse.training.llm.num_layer
    seq_tile_size = dse.training.llm.seq_tile_size
    
elif dse.training.WhichOneof('workload_variant') == 'dlrm':
    mlp_dim = dse.training.dlrm.mlp_dim
    bottom_num_mlp = dse.training.dlrm.bottom_num_mlp
    top_num_mlp = dse.training.dlrm.top_num_mlp
    pooled_row = dse.training.dlrm.pooled_row
    num_table = dse.training.dlrm.num_table
    emb = dse.training.dlrm.emb
    num_layer = 1

elif dse.training.WhichOneof('workload_variant') == 'hpl':
    n = dse.training.hpl.n
    b = dse.training.hpl.b
    num_layer = 1
    
elif dse.training.WhichOneof('workload_variant') == 'other':
    pass
    
else:
    raise Exception('Wrong!')


global_batch_size = dse.training.global_batch_size
micro_batch_size = dse.training.micro_batch_size
num_config = dse.training.num_config
optimization = dse.training.optimization
objective = dse.training.objective
util_threshold = dse.training.util_threshold
word = dse.training.word


if word == 0:
    raise Exception('Wrong!')


link_unit_price = dse.cost.link_unit_price
switch_unit_price = dse.cost.switch_unit_price
dram_unit_price = dse.cost.dram_unit_price


if dse.training.WhichOneof('workload_variant') == 'llm':
    Intermediate = hidden_dim * seq_len * word
    if num_head * head_dim != hidden_dim:
        raise Exception('Wrong!')

elif dse.training.WhichOneof('workload_variant') == 'dlrm' or dse.training.WhichOneof('workload_variant') == 'hpl':
    Intermediate = 0
     
elif dse.training.WhichOneof('workload_variant') == 'other':
    pass

else:
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
    







# get system info
Core = dse.system.accelerator.core
SRAM_Cap = dse.system.accelerator.sram_cap
VecWidth = dse.system.accelerator.systolic_width
StageWidth = dse.system.accelerator.systolic_height
Freq = dse.system.accelerator.freq
dram_bw = dse.system.accelerator.dram_bw
DRAM_Cap = dse.system.accelerator.dram_cap
num_chip = dse.system.num_chip
dram_bw_dse = dse.system.dram_bw_dse
net_bw_dse = dse.system.net_bw_dse
GFLOPS = 2*VecWidth*StageWidth*Core*Freq

    
if dse.system.WhichOneof('topology_variant') == 'single_chip': # single chip
    topology = []
    link_bw = []
    dimension = []
    
    a2a_bw_factor = []
    a2a_msg_factor = []

elif dse.system.WhichOneof('topology_variant') == 'sw': # 1D SW
    topology = [BasicTopology.SW.value]
    link_bw = [dse.system.sw.link_bw_x]
    dimension = [dse.system.sw.x]
    
    a2a_bw_factor = [dse.system.sw.x]
    a2a_msg_factor = [dse.system.sw.x**2 / 4]
    
elif dse.system.WhichOneof('topology_variant') == 'fc': # 1D FC
    topology = [BasicTopology.FC.value]
    link_bw = [dse.system.fc.link_bw_x]
    dimension = [dse.system.fc.x]
    
    a2a_bw_factor = [dse.system.fc.x**2 / 4]
    a2a_msg_factor = [dse.system.fc.x**2 / 4]
  
elif dse.system.WhichOneof('topology_variant') == 'r': # 1D Ring
    topology = [BasicTopology.R.value]
    link_bw = [dse.system.r.link_bw_x]
    dimension = [dse.system.r.x]    
    
    a2a_bw_factor = [2]
    a2a_msg_factor = [dse.system.r.x**2 / 4]
  
elif dse.system.WhichOneof('topology_variant') == 'r_r': # 2D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.r_r.link_bw_x, dse.system.r_r.link_bw_y]
    dimension = [dse.system.r_r.x, dse.system.r_r.y]

    a2a_bw_factor = [dse.system.r_r.x, dse.system.r_r.y]
    a2a_msg_factor = [num_chip**2 / 4, num_chip**2 / 4]
    
    p2p_bw_factor = 1
    p2p_msg_factor = 1
    bcast_bw_factor = 1
    bcast_msg_factor = (dse.system.r_r.y-1)/2
    
elif dse.system.WhichOneof('topology_variant') == 'fc_fc': # 2D Dragonfly
    topology = [BasicTopology.FC.value, BasicTopology.FC.value]
    link_bw = [dse.system.fc_fc.link_bw_x, dse.system.fc_fc.link_bw_y]
    dimension = [dse.system.fc_fc.x, dse.system.fc_fc.y]
    
    a2a_bw_factor = [dse.system.fc_fc.x**2 / 4, dse.system.fc_fc.y**2 / 4]
    a2a_msg_factor = [dse.system.fc_fc.x**2 / 4, num_chip**2 / 4]
    
    p2p_bw_factor = 1
    p2p_msg_factor = 1
    bcast_bw_factor = 1
    bcast_msg_factor = 1/2
    
elif dse.system.WhichOneof('topology_variant') == 'r_sw': # 2D DGX-1
    topology = [BasicTopology.R.value, BasicTopology.SW.value]
    link_bw = [dse.system.r_sw.link_bw_x, dse.system.r_sw.link_bw_y]
    dimension = [dse.system.r_sw.x, dse.system.r_sw.y]
    
    a2a_bw_factor = [2, dse.system.r_sw.y]
    a2a_msg_factor = [dse.system.r_sw.x**2 / 4, num_chip**2 / 4]
    
    p2p_bw_factor = 1
    p2p_msg_factor = 1
    bcast_bw_factor = 1
    bcast_msg_factor = 1/2
    
elif dse.system.WhichOneof('topology_variant') == 'sw_sw': # 2D DGX-2
    topology = [BasicTopology.SW.value, BasicTopology.SW.value]
    link_bw = [dse.system.sw_sw.link_bw_x, dse.system.sw_sw.link_bw_y]
    dimension = [dse.system.sw_sw.x, dse.system.sw_sw.y]
    
    a2a_bw_factor = [dse.system.sw_sw.x, dse.system.sw_sw.y]
    a2a_msg_factor = [dse.system.sw_sw.x**2 / 4, num_chip**2 / 4]
    
    p2p_bw_factor = 1
    p2p_msg_factor = 1
    bcast_bw_factor = 1
    bcast_msg_factor = 1/2
    
elif dse.system.WhichOneof('topology_variant') == 'r_fc': # Ring-FC
    topology = [BasicTopology.R.value, BasicTopology.FC.value]
    link_bw = [dse.system.r_fc.link_bw_x, dse.system.r_fc.link_bw_y]
    dimension = [dse.system.r_fc.x, dse.system.r_fc.y]
    
    a2a_bw_factor = [2, dse.system.r_fc.y**2 / 4]
    a2a_msg_factor = [dse.system.r_fc.x**2 / 4, num_chip**2 / 4]
    
    p2p_bw_factor = 1
    p2p_msg_factor = 1
    bcast_bw_factor = 1
    bcast_msg_factor = 1/2
    
elif dse.system.WhichOneof('topology_variant') == 'r_r_r': # 3D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.r_r_r.link_bw_x, dse.system.r_r_r.link_bw_y, dse.system.r_r_r.link_bw_z]
    dimension = [dse.system.r_r_r.x, dse.system.r_r_r.y, dse.system.r_r_r.z]
    
    a2a_bw_factor = [dse.system.r_r_r.x, dse.system.r_r_r.y, dse.system.r_r_r.z]
    a2a_msg_factor = [num_chip**2 / 4, num_chip**2 / 4, num_chip**2 / 4]
    
else:
    raise Exception('Wrong!')


model = gp.Model()
model.params.NonConvex = 2
model.Params.Threads = 140
model.params.MIPGap = 1e-50
model.params.TimeLimit = 36000






# TP/PP/DP
TP = model.addVar(name='TP', vtype=gp.GRB.INTEGER)
PP = model.addVar(name='PP', vtype=gp.GRB.INTEGER)
DP = model.addVar(name='DP', vtype=gp.GRB.INTEGER)


if dse.training.WhichOneof('workload_variant') == 'hpl':
    Shape = model.addMVar(len(topology), name='Shape', vtype=gp.GRB.INTEGER, lb=0)
    for i in range(len(topology)):
        model.addConstr(Shape[i] == dimension[i])
    
    model.addConstr(TP == 1)
    model.addConstr(PP == 1)
    model.addConstr(DP == num_chip)

    Link_BW = model.addMVar(len(topology), name='Link_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
    if net_bw_dse == True: # DSE
        for i in range(len(topology)):
            model.addConstr(Link_BW[i] <= link_bw[i])
    else:
        for i in range(len(topology)):
            model.addConstr(Link_BW[i] == link_bw[i])
            
elif dse.training.WhichOneof('workload_variant') == 'dlrm':
    Shape = model.addMVar(len(topology), name='Shape', vtype=gp.GRB.INTEGER, lb=0)
    for i in range(len(topology)):
        model.addConstr(Shape[i] == dimension[i])
    
    model.addConstr(TP == 1)
    model.addConstr(PP == 1)
    model.addConstr(DP == num_chip)   

    Link_BW = model.addMVar(len(topology), name='Link_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
    if net_bw_dse == True: # DSE
        for i in range(len(topology)):
            model.addConstr(Link_BW[i] <= link_bw[i])
    else:
        for i in range(len(topology)):
            model.addConstr(Link_BW[i] == link_bw[i])        

elif dse.training.WhichOneof('workload_variant') == 'llm':
    if len(topology) == 0: # single chip
        model.addConstr(TP == 1)
        model.addConstr(PP == 1)
        model.addConstr(DP == 1)   
        
        Link_BW_TP = model.addVar(name='Link_BW_TP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW_PP = model.addVar(name='Link_BW_PP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW_DP = model.addVar(name='Link_BW_PP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW = model.addVar(name='Link_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
        model.addConstr(Link_BW == sys.maxsize)
        
        model.addConstr(Link_BW_TP == Link_BW)
        model.addConstr(Link_BW_PP == Link_BW)
        model.addConstr(Link_BW_DP == Link_BW)
        
    elif len(topology) == 1: # 1D
        
        Shape = model.addMVar(1, name='Shape', vtype=gp.GRB.INTEGER, lb=0)
        model.addConstr(Shape[0] == num_chip)
        
        model.addConstr(TP == num_chip)
        model.addConstr(PP == 1)
        model.addConstr(DP == 1)   
        
        Link_BW = model.addMVar(1, name='Link_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
        if net_bw_dse == True: # DSE
            model.addConstr(Link_BW[0] <= link_bw[0])
        else:
            model.addConstr(Link_BW[0] == link_bw[0])
        
        Link_BW_TP = model.addVar(name='Link_BW_TP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW_PP = model.addVar(name='Link_BW_PP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW_DP = model.addVar(name='Link_BW_DP', vtype=gp.GRB.CONTINUOUS, lb=0)
        model.addConstr(Link_BW_TP == Link_BW[0])
        model.addConstr(Link_BW_PP == sys.maxsize)
        model.addConstr(Link_BW_DP == sys.maxsize)
            
    elif len(topology) == 2: # 2D

        Shape = model.addMVar(2, name='Shape', vtype=gp.GRB.INTEGER, lb=0)
        if dimension[0] == 0: # DSE on topology dimensions
            pass
        else:
            model.addConstr(Shape[0] == dimension[0])
            model.addConstr(Shape[1] == dimension[1])
        model.addConstr(Shape[0] * Shape[1] == num_chip)
        
        model.addConstr(TP == Shape[0])
        model.addConstr(PP == Shape[1])
        model.addConstr(DP == 1)

        Link_BW = model.addMVar(2, name='Link_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
        if net_bw_dse == True: # DSE
            model.addConstr(Link_BW[0] <= link_bw[0])
            model.addConstr(Link_BW[1] <= link_bw[1])
        else:
            model.addConstr(Link_BW[0] == link_bw[0])
            model.addConstr(Link_BW[1] == link_bw[1])
        
        Link_BW_TP = model.addVar(name='Link_BW_TP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW_PP = model.addVar(name='Link_BW_PP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW_DP = model.addVar(name='Link_BW_DP', vtype=gp.GRB.CONTINUOUS, lb=0)
        model.addConstr(Link_BW_TP == Link_BW[0])
        model.addConstr(Link_BW_PP == Link_BW[1])
        model.addConstr(Link_BW_DP == sys.maxsize)

    elif len(topology) == 3: # 3D
        Shape = model.addMVar(3, name='Shape', vtype=gp.GRB.INTEGER, lb=0)
        if dimension[0] == 0: # DSE on topology dimensions
            pass
        else:
            model.addConstr(Shape[0] == dimension[0])
            model.addConstr(Shape[1] == dimension[1])
            model.addConstr(Shape[2] == dimension[2])
        
        aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
        model.addConstr(aaa == Shape[0] * Shape[1])
        model.addConstr(aaa * Shape[2] == num_chip)
        
        model.addConstr(TP == Shape[0])
        model.addConstr(PP == Shape[1])
        model.addConstr(DP == Shape[2])
        
        Link_BW = model.addMVar(3, name='Link_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
        if net_bw_dse == True: # DSE
            model.addConstr(Link_BW[0] <= link_bw[0])
            model.addConstr(Link_BW[1] <= link_bw[1])
            model.addConstr(Link_BW[2] <= link_bw[2])
        else:
            model.addConstr(Link_BW[0] == link_bw[0])
            model.addConstr(Link_BW[1] == link_bw[1])
            model.addConstr(Link_BW[2] == link_bw[2])

        Link_BW_TP = model.addVar(name='Link_BW_TP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW_PP = model.addVar(name='Link_BW_PP', vtype=gp.GRB.CONTINUOUS, lb=0)
        Link_BW_DP = model.addVar(name='Link_BW_DP', vtype=gp.GRB.CONTINUOUS, lb=0)
        model.addConstr(Link_BW_TP == Link_BW[0])
        model.addConstr(Link_BW_PP == Link_BW[1])
        model.addConstr(Link_BW_DP == Link_BW[2])

    else:
        raise Exception('Wrong!')

elif dse.training.WhichOneof('workload_variant') == 'other':
    pass
    
else:
    raise Exception('Wrong!')






DRAM_BW = model.addVar(name='DRAM_BW', vtype=gp.GRB.CONTINUOUS, lb=0)
if dram_bw_dse == True: # DSE
    model.addConstr(DRAM_BW <= dram_bw)
else:
    model.addConstr(DRAM_BW == dram_bw)











# sharding/tiling kernel
tile_size = model.addVar(name='tile_size', vtype=gp.GRB.INTEGER, lb=1)
num_tile = model.addVar(name='num_tile', vtype=gp.GRB.INTEGER, lb=0)

if dse.training.WhichOneof('workload_variant') == 'llm':
    model.addConstr(tile_size * num_tile == seq_len)
    if seq_tile_size == 0: # DSE
        pass
    else:
        model.addConstr(tile_size == seq_tile_size)

elif dse.training.WhichOneof('workload_variant') == 'dlrm' or dse.training.WhichOneof('workload_variant') == 'hpl':
    model.addConstr(num_tile == 1)
     
elif dse.training.WhichOneof('workload_variant') == 'other':
    pass

else:
    raise Exception('Wrong!')




shard_M = model.addMVar(num_kernel, name='shard_M', vtype=gp.GRB.INTEGER, lb=0)
shard_K = model.addMVar(num_kernel, name='shard_K', vtype=gp.GRB.INTEGER, lb=0)
shard_N = model.addMVar(num_kernel, name='shard_N', vtype=gp.GRB.INTEGER, lb=0)

for i in range(num_kernel):
    if sharding[i] == Dim.OUTER_DIM.value or sharding[i] == Dim.M_DIM.value:
        if tiling[i] == Dim.OUTER_DIM.value or tiling[i] == Dim.M_DIM.value:
            raise Exception('Wrong!')
        
        elif tiling[i] == Dim.K_DIM.value:
            model.addConstr(shard_M[i] * TP >= M[i])
            model.addConstr(shard_K[i] * num_tile >= K[i])
            model.addConstr(shard_N[i] >= N[i])
            
        elif tiling[i] == Dim.N_DIM.value:
            model.addConstr(shard_M[i] * TP >= M[i])
            model.addConstr(shard_K[i] >= K[i])
            model.addConstr(shard_N[i] * num_tile >= N[i])
        
        elif tiling[i] == Dim.NO_DIM.value:    
            model.addConstr(shard_M[i] * TP >= M[i])
            model.addConstr(shard_K[i] >= K[i])
            model.addConstr(shard_N[i] >= N[i])
            
        else:
            raise Exception('Wrong!')
 
    elif sharding[i] == Dim.K_DIM.value:
        if tiling[i] == Dim.OUTER_DIM.value or tiling[i] == Dim.M_DIM.value:
            model.addConstr(shard_M[i] * num_tile >= M[i])
            model.addConstr(shard_K[i] * TP >= K[i])
            model.addConstr(shard_N[i] >= N[i])
        
        elif tiling[i] == Dim.K_DIM.value:
            raise Exception('Wrong!')
            
        elif tiling[i] == Dim.N_DIM.value:
            model.addConstr(shard_M[i] >= M[i])
            model.addConstr(shard_K[i] * TP >= K[i])
            model.addConstr(shard_N[i] * num_tile >= N[i])
            
        elif tiling[i] == Dim.NO_DIM.value:    
            model.addConstr(shard_M[i] >= M[i])
            model.addConstr(shard_K[i] * TP >= K[i])
            model.addConstr(shard_N[i] >= N[i])
        
        else:
            raise Exception('Wrong!')
        
    elif sharding[i] == Dim.N_DIM.value:
        if tiling[i] == Dim.OUTER_DIM.value or tiling[i] == Dim.M_DIM.value:
            model.addConstr(shard_M[i] * num_tile >= M[i])
            model.addConstr(shard_K[i] >= K[i])
            model.addConstr(shard_N[i] * TP >= N[i])
        
        elif tiling[i] == Dim.K_DIM.value:
            model.addConstr(shard_M[i] >= M[i])
            model.addConstr(shard_K[i] * num_tile >= K[i])
            model.addConstr(shard_N[i] * TP >= N[i])
            
        elif tiling[i] == Dim.N_DIM.value:
            raise Exception('Wrong!')
            
        elif tiling[i] == Dim.NO_DIM.value:    
            model.addConstr(shard_M[i] >= M[i])
            model.addConstr(shard_K[i] >= K[i])
            model.addConstr(shard_N[i] * TP >= N[i])
            
        else:
            raise Exception('Wrong!')
        
    elif sharding[i] == Dim.NO_DIM.value:
        if tiling[i] == Dim.OUTER_DIM.value or tiling[i] == Dim.M_DIM.value:
            model.addConstr(shard_M[i] * num_tile >= M[i])
            model.addConstr(shard_K[i] >= K[i])
            model.addConstr(shard_N[i] >= N[i])
        
        elif tiling[i] == Dim.K_DIM.value:
            model.addConstr(shard_M[i] >= M[i])
            model.addConstr(shard_K[i] * num_tile >= K[i])
            model.addConstr(shard_N[i] >= N[i])
            
        elif tiling[i] == Dim.N_DIM.value:
            model.addConstr(shard_M[i] >= M[i])
            model.addConstr(shard_K[i] >= K[i])
            model.addConstr(shard_N[i] * num_tile >= N[i])
        
        elif tiling[i] == Dim.NO_DIM.value:    
            model.addConstr(shard_M[i] >= M[i])
            model.addConstr(shard_K[i] >= K[i])
            model.addConstr(shard_N[i] >= N[i])
        
        else:
            raise Exception('Wrong!')
        
    else:
        raise Exception('Wrong!')



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
ALL_REDUCE_communication_size = model.addMVar(num_kernel, name='ALL_REDUCE_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)
ALL_REDUCE_PERIODIC_communication_size = model.addMVar(num_kernel, name='ALL_REDUCE_PERIODIC_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)
ALL_TO_ALL_communication_size = model.addMVar(num_kernel, name='ALL_TO_ALL_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)
POINT_TO_POINT_communication_size = model.addMVar(num_kernel, name='POINT_TO_POINT_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)
BROADCAST_communication_size = model.addMVar(num_kernel, name='BROADCAST_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)


Micro_Batch_Size = model.addVar(name='Micro_Batch_Size', vtype=gp.GRB.INTEGER, lb=1)
if micro_batch_size == 0: # DSE
    pass
else:
    model.addConstr(Micro_Batch_Size == micro_batch_size)
    
for i in range(num_kernel):
    if node_communication_type[i] == Communication.ALL_REDUCE.value:
        model.addConstr(ALL_REDUCE_communication_size[i] == shard_M[i] * shard_N[i] * word)

    elif node_communication_type[i] == Communication.ALL_REDUCE_PERIODIC.value:
        aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
        bbb = model.addVar(vtype=gp.GRB.CONTINUOUS)
        ccc = model.addVar(vtype=gp.GRB.CONTINUOUS)
        model.addConstr(aaa == shard_M[i] * shard_N[i])
        model.addConstr(bbb == PP * DP)
        model.addConstr(ccc == Micro_Batch_Size * bbb)
        model.addConstr(ALL_REDUCE_PERIODIC_communication_size[i] == aaa * word * ccc / global_batch_size)
        
    elif node_communication_type[i] == Communication.ALL_TO_ALL.value:
        model.addConstr(ALL_TO_ALL_communication_size[i] == node_communication_size[i])
    
    elif node_communication_type[i] == Communication.POINT_TO_POINT.value:
        model.addConstr(POINT_TO_POINT_communication_size[i] == node_communication_size[i])
    
    elif node_communication_type[i] == Communication.BROADCAST.value:
        model.addConstr(BROADCAST_communication_size[i] == node_communication_size[i])
        
    else:
        model.addConstr(ALL_REDUCE_communication_size[i] == 0)
        model.addConstr(ALL_REDUCE_PERIODIC_communication_size[i] == 0)
        model.addConstr(ALL_TO_ALL_communication_size[i] == 0)
        model.addConstr(POINT_TO_POINT_communication_size[i] == 0)
        model.addConstr(BROADCAST_communication_size[i] == 0)





if num_config == 0: # not specified
    C = num_kernel
else:
    C = num_config
    
    
Config = model.addMVar(num_kernel, name='Config', vtype=gp.GRB.INTEGER, lb=0)
Ab_onchip = model.addMVar((num_edge, C), name='Ab_onchip', vtype=gp.GRB.BINARY) # on-chip
Ab_dram = model.addMVar((num_edge, C), name='Ab_dram', vtype=gp.GRB.BINARY) # to/from DRAM
Ac = model.addMVar((num_kernel, C), name='Ac', vtype=gp.GRB.BINARY)
Ad = model.addMVar((num_weight, C), name='Ad', vtype=gp.GRB.BINARY)



if optimization == Optimization.KERNEL_BY_KERNEL.value:
    for i in range(len(configs)):
        model.addConstr(Config[i] == i) # tuning nobe
        
elif optimization == Optimization.FLASHATTENTION.value:
    for i in range(len(configs)):
        if configs[i] == -1: # not specified
            pass
        else:
            model.addConstr(Config[i] == configs[i])

else:
    raise Exception('Wrong!')





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




if dse.training.sync_async == Sync_Async.NO_TRAINING.value:
    pass
    
else:
    mid = int(C/2)
    
    for i in range(num_kernel):
        if fwd_bwd[i] == FWD_BWD.FWD.value: # first half of configs given to fwd
            model.addConstr(Config[i] >= 0)
            model.addConstr(Config[i] <= mid-1)
            
        elif fwd_bwd[i] == FWD_BWD.BWD.value: # second half of configs given to bwd
            model.addConstr(Config[i] >= mid)
            model.addConstr(Config[i] <= C-1)
            
        else:
            raise Exception('Wrong!')
            
weight_tiling = model.addMVar(num_kernel, name='weight_tiling', vtype=gp.GRB.INTEGER, lb=1)   
if optimization == Optimization.KERNEL_BY_KERNEL.value:
    for i in range(C):
        model.addConstr(np.ones((num_kernel)) @ Ac[:, i] >= 1)
    model.addConstr(C == num_kernel)
        
elif optimization == Optimization.FLASHATTENTION.value:
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
shard_intermediate_buffer_size_depth_one = model.addMVar(num_edge, name='shard_intermediate_buffer_size_depth_one', vtype=gp.GRB.INTEGER, lb=0)
shard_initiation_buffer_size_depth_one = model.addMVar(num_weight, name='shard_initiation_buffer_size_depth_one', vtype=gp.GRB.INTEGER, lb=0)


for i in range(num_edge):
    model.addConstr(shard_intermediate_buffer_size_depth_original[i] >= shard_intermediate_buffer_size[i] * depth[i])
    model.addConstr(shard_intermediate_buffer_size_depth_two[i] >= shard_intermediate_buffer_size[i] * 2)
    model.addConstr(shard_intermediate_buffer_size_depth_one[i] >= shard_intermediate_buffer_size[i] * 1)

for i in range(num_weight):
    node_idx = node_dict[weight_dict[i]]
    
    model.addConstr(shard_initiation_buffer_size_depth_one[i] * weight_tiling[node_idx] >= shard_initiation_buffer_size[i] * 1)


SRAM_Per_Config_total = model.addMVar(C, name='SRAM_Per_Config_total', vtype=gp.GRB.CONTINUOUS, lb=0)
SRAM_Per_Config_intermediate_dram = model.addMVar(C, name='SRAM_Per_Config_intermediate_dram', vtype=gp.GRB.CONTINUOUS, lb=0)
SRAM_Per_Config_intermediate_onchip = model.addMVar(C, name='SRAM_Per_Config_intermediate_onchip', vtype=gp.GRB.CONTINUOUS, lb=0)
SRAM_Per_Config_initiation = model.addMVar(C, name='SRAM_Per_Config_initiation', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    if optimization == Optimization.KERNEL_BY_KERNEL.value:
        model.addConstr(SRAM_Per_Config_intermediate_dram[i] == shard_intermediate_buffer_size_depth_two @ Ab_dram[:, i])
        
    elif optimization == Optimization.FLASHATTENTION.value:
        model.addConstr(SRAM_Per_Config_intermediate_dram[i] == shard_intermediate_buffer_size_depth_two @ Ab_dram[:, i])
        
    else:
        raise Exception('Wrong!')
    
    model.addConstr(SRAM_Per_Config_intermediate_onchip[i] == shard_intermediate_buffer_size_depth_original @ Ab_onchip[:, i])
    model.addConstr(SRAM_Per_Config_initiation[i] == shard_initiation_buffer_size_depth_one @ Ad[:, i])
    model.addConstr(SRAM_Per_Config_total[i] == SRAM_Per_Config_intermediate_dram[i] + SRAM_Per_Config_intermediate_onchip[i] + SRAM_Per_Config_initiation[i])
    
    model.addConstr(SRAM_Per_Config_total[i] <= SRAM_Cap)



# dram cap
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


if dse.training.WhichOneof('workload_variant') == 'llm':
    model.addConstr((dram_bytes_initiation + dram_bytes_intermediate * Micro_Batch_Size) * num_layer <= DRAM_Cap * PP)

elif dse.training.WhichOneof('workload_variant') == 'dlrm' or dse.training.WhichOneof('workload_variant') == 'hpl':
    model.addConstr((dram_bytes_initiation + dram_bytes_intermediate * 1) * num_layer <= DRAM_Cap * PP)
     
elif dse.training.WhichOneof('workload_variant') == 'other':
    pass

else:
    raise Exception('Wrong!')






# record the weight tiling factor for kernel-by-kernel
if optimization == Optimization.KERNEL_BY_KERNEL.value:
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
            





Compute_Latency = model.addMVar(C, name='Compute_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    t1 = model.addMVar(num_kernel, vtype=gp.GRB.INTEGER, lb=0)
    t2 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    for j in range(num_kernel):
        model.addConstr(t1[j] == Cycle[j] * Ac[j, i])
    model.addConstr(t2 == gp.max_(t1[j] for j in range(num_kernel)))
    model.addConstr(Compute_Latency[i] == t2 * num_tile / Freq)


DRAM_bytes = model.addMVar(C, name='DRAM_bytes', vtype=gp.GRB.CONTINUOUS, lb=0)
DRAM_Latency = model.addMVar(C, name='DRAM_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    t1 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(t1 == shard_intermediate_buffer_size @ Ab_dram[:, i])
    
    if optimization == Optimization.KERNEL_BY_KERNEL.value:
        aaa = model.addVar(vtype=gp.GRB.INTEGER)
        model.addConstr(aaa == num_tile * weight_tiling_per_config[i])
        model.addConstr(DRAM_Latency[i] * DRAM_BW == t1 * aaa)
        model.addConstr(DRAM_bytes[i] == t1 * aaa)
    
    elif optimization == Optimization.FLASHATTENTION.value:
        model.addConstr(DRAM_Latency[i] * DRAM_BW == t1 * num_tile)
        model.addConstr(DRAM_bytes[i] == t1 * num_tile)
        
    else:
        raise Exception('Wrong!')
    
    
total_DRAM_bytes = model.addVar(name='total_DRAM_bytes', vtype=gp.GRB.CONTINUOUS)  
model.addConstr(total_DRAM_bytes == np.ones((C)) @ DRAM_bytes)








Network_Latency = model.addMVar(C, name='Network_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
total_Network_bytes = model.addVar(name='total_Network_bytes', vtype=gp.GRB.CONTINUOUS)

if dse.training.WhichOneof('workload_variant') == 'llm':
    # tensor parallelism all-reduce
    ALL_REDUCE_ratio = model.addVar(name='ALL_REDUCE_ratio', vtype=gp.GRB.CONTINUOUS) 
    if len(topology) == 0: # single chip
        model.addConstr(ALL_REDUCE_ratio == 0)

    elif 1 <= len(topology) <= 3: # 1D/2D/3D
        aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
        model.addConstr(aaa == TP * Link_BW_TP)
        if topology[0] == BasicTopology.R.value:
            model.addConstr(ALL_REDUCE_ratio * aaa * 2 == TP - 1)
        elif topology[0] == BasicTopology.FC.value:
            model.addConstr(ALL_REDUCE_ratio * aaa == 1)
        elif topology[0] == BasicTopology.SW.value:
            model.addConstr(ALL_REDUCE_ratio * aaa * 2 == TP - 1)
        else:
            raise Exception('Wrong!')

    else:
        raise Exception('Wrong!')

     
    Network_Bytes_ALL_REDUCE = model.addMVar(C, name='Network_Bytes_ALL_REDUCE', vtype=gp.GRB.CONTINUOUS, lb=0)
    Network_Latency_ALL_REDUCE = model.addMVar(C, name='Network_Latency_ALL_REDUCE', vtype=gp.GRB.CONTINUOUS, lb=0)
    for i in range(C):
        t1 = model.addVar(vtype=gp.GRB.CONTINUOUS)
        t2 = model.addVar(vtype=gp.GRB.CONTINUOUS)
        model.addConstr(t1 == Ac[:, i] @ ALL_REDUCE_communication_size)
        model.addConstr(t2 == ALL_REDUCE_ratio * num_tile)
        model.addConstr(Network_Latency_ALL_REDUCE[i] == 2 * t1 * t2) # reduce-scatter/all-gather
        model.addConstr(Network_Bytes_ALL_REDUCE[i] == Network_Latency_ALL_REDUCE[i] * Link_BW_TP)




    # data parallelism all-reduce
    ALL_REDUCE_PERIODIC_ratio = model.addVar(name='ALL_REDUCE_PERIODIC_ratio', vtype=gp.GRB.CONTINUOUS) 
    if 0 <= len(topology) <= 2: # single chip/1D/2D
        model.addConstr(ALL_REDUCE_PERIODIC_ratio == 0)

    elif len(topology) == 3: # 3D
        aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
        model.addConstr(aaa == DP * Link_BW_DP)
        if topology[0] == BasicTopology.R.value:
            model.addConstr(ALL_REDUCE_PERIODIC_ratio * aaa * 2 == DP - 1)
        elif topology[0] == BasicTopology.FC.value:
            model.addConstr(ALL_REDUCE_PERIODIC_ratio * aaa == 1)
        elif topology[0] == BasicTopology.SW.value:
            model.addConstr(ALL_REDUCE_PERIODIC_ratio * aaa * 2 == DP - 1)
        else:
            raise Exception('Wrong!')
            
    else:
        raise Exception('Wrong!')


    Network_Bytes_ALL_REDUCE_PERIODIC = model.addMVar(C, name='Network_Bytes_ALL_REDUCE_PERIODIC', vtype=gp.GRB.CONTINUOUS, lb=0)
    Network_Latency_ALL_REDUCE_PERIODIC = model.addMVar(C, name='Network_Latency_ALL_REDUCE_PERIODIC', vtype=gp.GRB.CONTINUOUS, lb=0)
    for i in range(C):
        t1 = model.addVar(vtype=gp.GRB.CONTINUOUS)
        t2 = model.addVar(vtype=gp.GRB.CONTINUOUS)
        model.addConstr(t1 == Ac[:, i] @ ALL_REDUCE_PERIODIC_communication_size)
        model.addConstr(t2 == ALL_REDUCE_PERIODIC_ratio)
        model.addConstr(Network_Latency_ALL_REDUCE_PERIODIC[i] == 2 * t1 * t2) # reduce-scatter/all-gather
        model.addConstr(Network_Bytes_ALL_REDUCE_PERIODIC[i] == Network_Latency_ALL_REDUCE_PERIODIC[i] * Link_BW_DP)




    # total network latency from data/tensor parallelism
    for i in range(C):
        model.addConstr(Network_Latency[i] == Network_Latency_ALL_REDUCE[i] + Network_Latency_ALL_REDUCE_PERIODIC[i])
        
        

    # total network bytes from data/tensor parallelism
    model.addConstr(total_Network_bytes == np.ones((C)) @ Network_Bytes_ALL_REDUCE + np.ones((C)) @ Network_Bytes_ALL_REDUCE_PERIODIC)
    
    
    
    
    
    
    # pipeline parallelism point-to-point
    aaa = model.addVar(vtype=gp.GRB.BINARY)
    intermediate = model.addVar(name='intermediate', vtype=gp.GRB.CONTINUOUS, lb=0)
    model.addConstr((aaa == 1) >> (PP == 1))
    model.addConstr((aaa == 0) >> (PP >= 2))
    model.addConstr((aaa == 1) >> (intermediate == 0))
    model.addConstr((aaa == 0) >> (intermediate == Intermediate))

    p2p_latency = model.addVar(name='p2p_latency', vtype=gp.GRB.CONTINUOUS)
    model.addConstr(p2p_latency * Link_BW_PP == intermediate)
    
    

elif dse.training.WhichOneof('workload_variant') == 'dlrm':
    accumulated_shape = model.addMVar(len(topology), vtype=gp.GRB.CONTINUOUS, lb=0)
    for i in range(len(topology)):
        if i == 0:
            model.addConstr(accumulated_shape[i] == Shape[i])
        else:
            model.addConstr(accumulated_shape[i] == accumulated_shape[i-1] * Shape[i])
    
    
    
    Network_Latency_ALL_TO_ALL_tmp = model.addMVar((C, len(topology)), name='Network_Latency_ALL_TO_ALL_tmp', vtype=gp.GRB.CONTINUOUS, lb=0)
    Network_Latency_ALL_TO_ALL = model.addMVar(C, name='Network_Latency_ALL_TO_ALL', vtype=gp.GRB.CONTINUOUS, lb=0)
    Network_Bytes_ALL_TO_ALL_tmp = model.addMVar((C, len(topology)), name='Network_Bytes_ALL_TO_ALL_tmp', vtype=gp.GRB.CONTINUOUS, lb=0)
    Network_Bytes_ALL_TO_ALL = model.addMVar(C, name='Network_Bytes_ALL_TO_ALL', vtype=gp.GRB.CONTINUOUS, lb=0)
    
    for i in range(C):
        for j in range(len(topology)):
            aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
            model.addConstr(aaa == Ac[:, i] @ ALL_TO_ALL_communication_size) # per-chip bytes
            
            model.addConstr(Network_Latency_ALL_TO_ALL_tmp[i, j] * Link_BW[j] * a2a_bw_factor[j] == aaa * a2a_msg_factor[j])
            model.addConstr(Network_Bytes_ALL_TO_ALL_tmp[i, j] == Network_Latency_ALL_TO_ALL_tmp[i, j] * Link_BW[j])
     
    
    for i in range(C):
        model.addConstr(Network_Latency_ALL_TO_ALL[i] == gp.max_(Network_Latency_ALL_TO_ALL_tmp[i, j] for j in range(len(topology))))
        model.addConstr(Network_Bytes_ALL_TO_ALL[i] == gp.max_(Network_Bytes_ALL_TO_ALL_tmp[i, j] for j in range(len(topology))))
        
    for i in range(C):
        model.addConstr(Network_Latency[i] == Network_Latency_ALL_TO_ALL[i])
    model.addConstr(total_Network_bytes == np.ones((C)) @ Network_Bytes_ALL_TO_ALL)
        
    
    
elif dse.training.WhichOneof('workload_variant') == 'hpl':
    Network_Latency_POINT_TO_POINT = model.addMVar(C, name='Network_Latency_POINT_TO_POINT', vtype=gp.GRB.CONTINUOUS, lb=0)
    Network_Bytes_POINT_TO_POINT = model.addMVar(C, name='Network_Bytes_POINT_TO_POINT', vtype=gp.GRB.CONTINUOUS, lb=0)
    Network_Latency_BROADCAST = model.addMVar(C, name='Network_Latency_BROADCAST', vtype=gp.GRB.CONTINUOUS, lb=0)
    Network_Bytes_BROADCAST = model.addMVar(C, name='Network_Bytes_BROADCAST', vtype=gp.GRB.CONTINUOUS, lb=0)
    
    for i in range(C):
        # X dim
        aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
        model.addConstr(aaa == Ac[:, i] @ POINT_TO_POINT_communication_size)
        
        # Y dim
        bbb = model.addVar(vtype=gp.GRB.CONTINUOUS)
        model.addConstr(bbb == Ac[:, i] @ BROADCAST_communication_size)
        
        model.addConstr(Network_Latency_POINT_TO_POINT[i] * Link_BW[0] * p2p_bw_factor == aaa * p2p_msg_factor)
        model.addConstr(Network_Bytes_POINT_TO_POINT[i] == Link_BW[0] * Network_Latency_POINT_TO_POINT[i])
        model.addConstr(Network_Latency_BROADCAST[i] * Link_BW[1] * bcast_bw_factor == bbb * bcast_msg_factor)
        model.addConstr(Network_Bytes_BROADCAST[i] == Link_BW[1] * Network_Latency_BROADCAST[i])
        
    for i in range(C):
        model.addConstr(Network_Latency[i] == gp.max_(Network_Latency_POINT_TO_POINT[i], Network_Latency_BROADCAST[i]))
    model.addConstr(total_Network_bytes == np.ones((C)) @ Network_Bytes_POINT_TO_POINT + np.ones((C)) @ Network_Bytes_BROADCAST)
    model.addConstr(total_Network_bytes == np.ones((C)) @ Network_Bytes_POINT_TO_POINT + np.ones((C)) @ Network_Bytes_BROADCAST)

elif dse.training.WhichOneof('workload_variant') == 'other':
    pass
    
else:
    raise Exception('Wrong!')











# weight loading overheads
Setup_Latency = model.addMVar(C, name='Setup_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
    time_1 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    time_2 = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(aaa == DRAM_BW * Micro_Batch_Size)
    model.addConstr(time_1 * aaa == shard_initiation_buffer_size @ Ad[:, i])
    model.addConstr(time_2 * DRAM_BW == memory_size @ Ac[:, i])
    model.addConstr(Setup_Latency[i] == time_1 + time_2)



Latency_wo_setup = model.addMVar(C, name='Latency_wo_setup', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    model.addConstr(Latency_wo_setup[i] == gp.max_(Compute_Latency[i], DRAM_Latency[i], Network_Latency[i]))

Per_Config_II = model.addMVar(C, name='Per_Config_II', vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(C):
    model.addConstr(Per_Config_II[i] == Latency_wo_setup[i] + Setup_Latency[i])
    




II = model.addVar(name='II', vtype=gp.GRB.CONTINUOUS)

if dse.training.WhichOneof('workload_variant') == 'llm':
    aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(aaa == np.ones((C)) @ Per_Config_II)
    model.addConstr(II == gp.max_(aaa, p2p_latency))

elif dse.training.WhichOneof('workload_variant') == 'dlrm' or dse.training.WhichOneof('workload_variant') == 'hpl':
    model.addConstr(II == np.ones((C)) @ Per_Config_II)
    
elif dse.training.WhichOneof('workload_variant') == 'other':
    pass
    
else:
    raise Exception('Wrong!')









if util_threshold == 0.0:
    pass
else:
    FLOP_per_layer = 0.0
    for i in range(len(M)):
        if kernel_type[i] == KernelType.SIMD.value:
            FLOP_per_layer += M[i] * K[i] * N[i]
        else:
            FLOP_per_layer += 2 * M[i] * K[i] * N[i]
            
    model.addConstr(FLOP_per_layer >= util_threshold * GFLOPS * II * TP)




# cost
DRAM_cost = model.addVar(name='DRAM_cost', vtype=gp.GRB.CONTINUOUS, lb=0)
model.addConstr(DRAM_cost == num_chip * DRAM_BW * dram_unit_price)



accumulated_shape = model.addMVar(len(topology), vtype=gp.GRB.CONTINUOUS, lb=0)
for i in range(len(topology)):
    if i == 0:
        model.addConstr(accumulated_shape[i] == Shape[i])
    else:
        model.addConstr(accumulated_shape[i] == accumulated_shape[i-1] * Shape[i])
    


  
Link_Switch_cost = model.addMVar(len(topology), name='Link_Switch_cost', vtype=gp.GRB.CONTINUOUS, lb=0)
if len(topology) == 0:
    pass
else:
    for i in range(len(topology)):
        if i == 0:
            if topology[i] == BasicTopology.R.value:
                model.addConstr(Link_Switch_cost[i] == Shape[i] * Link_BW[i] * link_unit_price)
                
            elif topology[i] == BasicTopology.FC.value:
                aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
                model.addConstr(aaa == Shape[i] * (Shape[i] - 1))
                model.addConstr(Link_Switch_cost[i] == aaa / 2 * Link_BW[i] * link_unit_price)
                
            elif topology[i] == BasicTopology.SW.value:
                model.addConstr(Link_Switch_cost[i] == Shape[i] * Link_BW[i] * link_unit_price + Shape[i] * Link_BW[i] * switch_unit_price)
            
            else:
                raise Exception('Wrong!')
         
        else:
            if topology[i] == BasicTopology.R.value:
                model.addConstr(Link_Switch_cost[i] == Link_Switch_cost[i-1] * Shape[i] + accumulated_shape[i] * Link_BW[i] * link_unit_price)
                
            elif topology[i] == BasicTopology.FC.value:
                aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
                model.addConstr(aaa == Shape[i] * (Shape[i] - 1))
                model.addConstr(Link_Switch_cost[i] == Link_Switch_cost[i-1] * Shape[i] + aaa * Link_BW[i] / 2 * link_unit_price)
                
            elif topology[i] == BasicTopology.SW.value:
                model.addConstr(Link_Switch_cost[i] == Link_Switch_cost[i-1] * Shape[i] + accumulated_shape[i] * Link_BW[i] * link_unit_price + accumulated_shape[i] * Link_BW[i] * switch_unit_price)
            
            else:
                raise Exception('Wrong!')
            
        
total_cost = model.addVar(name='total_cost', vtype=gp.GRB.CONTINUOUS, lb=0)
if len(topology) == 0:
    model.addConstr(total_cost == DRAM_cost)
else:
    aaa = model.addVar(vtype=gp.GRB.CONTINUOUS)
    model.addConstr(aaa == Link_Switch_cost[-1])
    model.addConstr(total_cost == DRAM_cost + aaa)




if objective == Objective.PERFORMANCE.value:
    model.setObjective(II, gp.GRB.MINIMIZE)
    
elif objective == Objective.COST.value:
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    
else:
    raise Exception('Wrong!')
    
model.optimize()



# get values from gurobi
shard_M = []
shard_K = []
shard_N = []
shard_intermediate_buffer_size = []
shard_initiation_buffer_size = []
Config = []
Link_BW = []
Per_Config_II = []
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
    if v.varName.startswith('DP'):
        DP = v.X
    if v.varName.startswith('Config'):
        Config.append(v.X)
    if v.varName.startswith('total_cost'):
        total_cost = v.X  
    if v.varName.startswith('DRAM_BW'):
        DRAM_BW = v.X 
    if v.varName.startswith('Micro_Batch_Size'):
        Micro_Batch_Size = v.X 
    if v.varName.startswith('Link_BW['):
        Link_BW.append(v.X)
    if v.varName.startswith('num_config'):
        num_config = v.X
    if v.varName.startswith('Per_Config_II'):
        Per_Config_II.append(v.X)
        
        

FLOP = 0.0
for i in range(len(M)):
    if kernel_type[i] == KernelType.SIMD.value:
        FLOP += M[i] * K[i] * N[i]
    else:
        FLOP += 2 * M[i] * K[i] * N[i]
FLOP *= num_layer
        
layers_per_stage = num_layer / PP
II *= layers_per_stage





print('TP', TP)   
print('PP', PP)   
print('DP', DP)  
print('Micro_Batch_Size', Micro_Batch_Size)  
print('layers_per_stage', layers_per_stage) 
print('II', II)
        

if dse.training.sync_async == Sync_Async.NO_TRAINING.value:
    print('No training, same II')
    
else:
    print('num_config', C)
    
    II_fwd = 0
    II_bwd = 0
    for i in range(0, int(C/2)):
        II_fwd += Per_Config_II[i]
            
    for i in range(int(C/2), C):
        II_bwd += Per_Config_II[i]
        
    II_fwd *= layers_per_stage
    II_bwd *= layers_per_stage
    
    print('II_fwd', II_fwd)
    print('II_bwd', II_bwd)

    if dse.training.sync_async == Sync_Async.SYNC.value:
        II = (PP + Micro_Batch_Size - 1) / Micro_Batch_Size * (II_fwd + II_bwd)
        print('sync II:', II)
        
    elif dse.training.sync_async == Sync_Async.ASYNC.value:
        II = (Micro_Batch_Size-1) * max(II_fwd, II_bwd) + PP * (II_fwd+II_bwd)
        print('async:', II)
        
    else:
        raise Exception('Wrong!')
    
    




    
print('GFLOPS', GFLOPS) 
print('FLOP', FLOP)
print('total_cost', total_cost)
print('DRAM_BW', DRAM_BW)
print('Link_BW', Link_BW)
print('Samples/s', 1e9/II*DP)
print('util', DP*FLOP/II/GFLOPS/num_chip)
if total_DRAM_bytes == 0:
    print('oim infinity')
else:
    print('OI Memory', DP*FLOP / total_DRAM_bytes / layers_per_stage / num_chip)
if total_Network_bytes == 0:
    print('oin infinity')
else:
    print('OI network', DP*FLOP / total_Network_bytes / layers_per_stage / num_chip)




# update kernels
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.WhichOneof('kernel_variant') == 'gemm_input1_weight':
        kernel.gemm_input1_weight.shard_outer_M = int(shard_M[i])
        kernel.gemm_input1_weight.shard_K = int(shard_K[i])
        kernel.gemm_input1_weight.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
    
    elif kernel.WhichOneof('kernel_variant') == 'gemm_input1_input2':
        kernel.gemm_input1_input2.shard_outer_M = int(shard_M[i])
        kernel.gemm_input1_input2.shard_K = int(shard_K[i])
        kernel.gemm_input1_input2.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1':
        kernel.elementwise_input1.shard_outer_M = int(shard_M[i])
        kernel.elementwise_input1.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1_input2':
        kernel.elementwise_input1_input2.shard_outer_M = int(shard_M[i])
        kernel.elementwise_input1_input2.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
     
    else:
        raise Exception('Wrong!')
   
    i += 1

# update edges
i = 0
for connection in dse.dataflow_graph.connections:
    connection.shard_tensor_size = float(shard_intermediate_buffer_size[i])
    i += 1





# write to final binary
with open('./'+name+'/'+'dse_final.pb', "wb") as file:
    file.write(dse.SerializeToString())


# write to final text file
with open('./'+name+'/'+'dse_final.txt', "w") as file:
    text_format.PrintMessage(dse, file)



if dse.training.pydot:
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

