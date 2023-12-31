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
with open('./'+name+'/'+'dse.pb', "rb") as file:
    dse.ParseFromString(file.read())
    
    






# get dataflow graph info
kernel_name = []
output_tensor_size = []
kernel_type = []
outer = []
M = []
K = []
N = []
weight_tensor_size = []
input_tensor_1_size = []
input_tensor_2_size = []
tiling = []
node_dict = {}
i = 0
for kernel in dse.dataflow_graph.kernels:
    kernel_name.append(kernel.name)
    kernel_type.append(kernel.type)
    
    if kernel.WhichOneof('kernel_variant') == 'gemm_input1_weight':  
        outer.append(kernel.gemm_input1_weight.outer)
        M.append(kernel.gemm_input1_weight.M)
        K.append(kernel.gemm_input1_weight.K)
        N.append(kernel.gemm_input1_weight.N)
        
        input_tensor_1_size.append(kernel.gemm_input1_weight.input_tensor_size)
        input_tensor_2_size.append(-1.0)
        weight_tensor_size.append(kernel.gemm_input1_weight.weight_tensor_size)
        output_tensor_size.append(kernel.gemm_input1_weight.output_tensor_size)
        
        tiling.append(kernel.gemm_input1_weight.tiling)
        
    elif kernel.WhichOneof('kernel_variant') == 'gemm_input1_input2':
        outer.append(kernel.gemm_input1_input2.outer)
        M.append(kernel.gemm_input1_input2.M)
        K.append(kernel.gemm_input1_input2.K)
        N.append(kernel.gemm_input1_input2.N)
        
        input_tensor_1_size.append(kernel.gemm_input1_input2.input_tensor_1_size)
        input_tensor_2_size.append(kernel.gemm_input1_input2.input_tensor_2_size)
        weight_tensor_size.append(-1.0)
        output_tensor_size.append(kernel.gemm_input1_input2.output_tensor_size)
        
        tiling.append(kernel.gemm_input1_input2.tiling)
            
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1':
        outer.append(kernel.elementwise_input1.outer)
        M.append(kernel.elementwise_input1.M)
        K.append(1)
        N.append(kernel.elementwise_input1.N)
        
        input_tensor_1_size.append(kernel.elementwise_input1.input_tensor_size)
        input_tensor_2_size.append(-1.0)
        weight_tensor_size.append(-1.0)
        output_tensor_size.append(kernel.elementwise_input1.output_tensor_size)
        
        tiling.append(kernel.elementwise_input1.tiling)
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1_input2':
        outer.append(kernel.elementwise_input1_input2.outer)
        M.append(kernel.elementwise_input1_input2.M)
        K.append(1)
        N.append(kernel.elementwise_input1_input2.N)
        
        input_tensor_1_size.append(kernel.elementwise_input1_input2.input_tensor_1_size)
        input_tensor_2_size.append(kernel.elementwise_input1_input2.input_tensor_2_size)
        weight_tensor_size.append(-1.0)
        output_tensor_size.append(kernel.elementwise_input1_input2.output_tensor_size)
        
        tiling.append(kernel.elementwise_input1_input2.tiling)

    else:
        raise Exception('Wrong!')
    
    node_dict[kernel.id] = i
    i += 1


startIdx = [] # upstream node id
endIdx = [] # upstream node id
edge_dict = {}
i = 0
for connection in dse.dataflow_graph.connections:
    startIdx.append(connection.startIdx)
    endIdx.append(connection.endIdx)
    edge_dict[(connection.startIdx, connection.endIdx)] = i
    i += 1


# assign edge tensor size
tensor_size = []
for connection in dse.dataflow_graph.connections:
    if dse.training.skip_intermediate_buffer:
        connection.tensor_size = 0
    else:
        connection.tensor_size = output_tensor_size[node_dict[connection.startIdx]]
    
    tensor_size.append(connection.tensor_size)

num_kernel = len(kernel_name)
num_edge = len(startIdx)



# set buffer depth
kernel_topological_dict = {}
kernel_id_list = copy.deepcopy(list(node_dict.keys()))

edge_start_dict = {}
for connection in dse.dataflow_graph.connections:
    if connection.startIdx not in edge_start_dict.keys():
        edge_start_dict[connection.startIdx] = [connection.endIdx]
    else:
        edge_start_dict[connection.startIdx].append(connection.endIdx)

indegree = {}
for id in kernel_id_list:
    indegree[id] = 0
for start, end in edge_dict.keys():
    indegree[end] += 1


cnt = 0
while len(indegree.keys()) > 0:
    tmp = []
    for id in kernel_id_list:
        if id in indegree.keys() and indegree[id] == 0:
            kernel_topological_dict[id] = cnt
            del indegree[id]
            tmp.append(id)

    for id in tmp:
        if id in edge_start_dict.keys():
            for next_node_id in edge_start_dict[id]:
                indegree[next_node_id] -= 1
    cnt += 1




for kernel in dse.dataflow_graph.kernels:
    kernel.topological_number = kernel_topological_dict[kernel.id]

for connection in dse.dataflow_graph.connections:
    connection.buffer_depth = kernel_topological_dict[connection.endIdx] - kernel_topological_dict[connection.startIdx] + 1

startName = []
endName = []
i = 0
for connection in dse.dataflow_graph.connections:
    connection.startName = kernel_name[node_dict[startIdx[i]]]
    connection.endName = kernel_name[node_dict[endIdx[i]]]
    startName.append(connection.startName)
    endName.append(connection.endName)
    i += 1








    



class KernelType(Enum):
    SYSTOLIC = 1
    SIMD = 2



class Communication(Enum):
    NO_COMMUNICATION = 0
    ALL_REDUCE = 1
    ALL_TO_ALL = 2
    ALL_GATHER = 3
    ALL_REDUCE_PERIODIC = 4


class Dim(Enum):
    OUTER = 0
    M = 1
    K = 2
    N = 3
    NO_DIM = 4
    

class Tensor(Enum):
    RR = 0
    RS = 1
    SR = 2
    
    

for i in range(num_kernel):
    print(kernel_name[i], i)
    
for i in range(num_edge):
    print(startName[i], endName[i], i)





if dse.training.skip_inter_chip:
    sharding = []
    for i in range(5*num_kernel):
        sharding.append(Dim.NO_DIM.value)
        
    communication_size = []
    communication_type = []
    for i in range(num_kernel):
        communication_size.append(0)
        communication_type.append(Communication.NO_COMMUNICATION.value)

    edge_communication_size = []
    edge_communication_type = []
    for i in range(num_edge):
        edge_communication_size.append(0)
        edge_communication_type.append(Communication.NO_COMMUNICATION.value)
    
else:
    model = gp.Model()
    model.params.NonConvex = 2
    model.Params.Threads = 128
    model.params.MIPGap = 1e-10


    sharding = model.addMVar((num_kernel, 5), name='sharding', vtype=gp.GRB.BINARY) # outer,M,K,N,no sharding
    communication_type = model.addMVar((num_kernel), name='communication_type', vtype=gp.GRB.INTEGER, lb=Communication.NO_COMMUNICATION.value, ub=Communication.ALL_GATHER.value)
    communication_size = model.addMVar((num_kernel), name='communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)

    for i in range(num_kernel):
        if outer[i] == 1:
            model.addConstr(sharding[i, Dim.OUTER.value] == 0)
        if M[i] == 1:
            model.addConstr(sharding[i, Dim.M.value] == 0)
        if K[i] == 1:
            model.addConstr(sharding[i, Dim.K.value] == 0)
        if N[i] == 1:
            model.addConstr(sharding[i, Dim.N.value] == 0)
        model.addConstr(np.ones((5)) @ sharding[i, :] == 1)
            
        # don't shard the tile dim
        if tiling[i] == Dim.OUTER.value+1:
            model.addConstr(sharding[i, Dim.OUTER.value] == 0)
        elif tiling[i] == Dim.M.value+1:
            model.addConstr(sharding[i, Dim.M.value] == 0)
        elif tiling[i] == Dim.K.value+1:
            model.addConstr(sharding[i, Dim.K.value] == 0)
        elif tiling[i] == Dim.N.value+1:
            model.addConstr(sharding[i, Dim.N.value] == 0)
        else:
            raise Exception('Wrong!')
        
        

        if weight_tensor_size[i] == -1: # no weights
            pass
        else:
            model.addConstr(sharding[i, Dim.NO_DIM.value] == 0)



        # if K is sharded
        model.addConstr((sharding[i, Dim.K.value] == 1) >> (communication_type[i] == Communication.ALL_REDUCE.value))
        model.addConstr((sharding[i, Dim.K.value] == 1) >> (communication_size[i] == output_tensor_size[i]))

        # if K is not sharded
        model.addConstr((sharding[i, Dim.K.value] == 0) >> (communication_type[i] == Communication.NO_COMMUNICATION.value))
        model.addConstr((sharding[i, Dim.K.value] == 0) >> (communication_size[i] == 0))

        if outer[i] > 1:
            model.addConstr(sharding[i, Dim.OUTER.value] == 1)




        

    # RR, RS, SR
    upstream_sharding = model.addMVar((num_edge, 3), name='upstream_sharding', vtype=gp.GRB.BINARY)
    downstream_sharding = model.addMVar((num_edge, 3), name='downstream_sharding', vtype=gp.GRB.BINARY)
    for i in range(num_edge):
        model.addConstr(np.ones((3)) @ upstream_sharding[i, :] == 1)
        model.addConstr(np.ones((3)) @ downstream_sharding[i, :] == 1)
        
        upstream_node_idx = node_dict[startIdx[i]]
        downstream_node_idx = node_dict[endIdx[i]]
        
        
        
        
        # upsteam 
        model.addConstr((sharding[upstream_node_idx, Dim.OUTER.value] == 1) >> (upstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
        model.addConstr((sharding[upstream_node_idx, Dim.M.value] == 1) >> (upstream_sharding[i, Tensor.SR.value] == 1)) # shard M
        model.addConstr((sharding[upstream_node_idx, Dim.K.value] == 1) >> (upstream_sharding[i, Tensor.RR.value] == 1)) # shard K
        model.addConstr((sharding[upstream_node_idx, Dim.N.value] == 1) >> (upstream_sharding[i, Tensor.RS.value] == 1)) # shard N
        model.addConstr((sharding[upstream_node_idx, Dim.NO_DIM.value] == 1) >> (upstream_sharding[i, Tensor.RR.value] == 1)) # no sharding



        # downstream
        if tiling[downstream_node_idx] == Dim.K.value+1: # for weight update kernels
            model.addConstr((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
            model.addConstr((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
            model.addConstr((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1))
            model.addConstr((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
            model.addConstr((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1))
        
        else:
            if kernel_type[downstream_node_idx] == KernelType.SIMD.value:
                model.addConstr((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                model.addConstr((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard M
                model.addConstr((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard K
                model.addConstr((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                model.addConstr((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

            else:
                if weight_tensor_size[downstream_node_idx] != -1: # weight is present, this edge represents outer,K,N
                    model.addConstr((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                    model.addConstr((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard M
                    model.addConstr((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard K
                    model.addConstr((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                    model.addConstr((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                else: # weight is not present
                    if tensor_size[i] == input_tensor_1_size[downstream_node_idx]: # if the edge represent tensor 1
                        model.addConstr((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                        model.addConstr((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard M
                        model.addConstr((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard K
                        model.addConstr((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                        model.addConstr((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                    elif tensor_size[i] == input_tensor_2_size[downstream_node_idx]: # if the edge represent tensor 2
                        model.addConstr((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                        model.addConstr((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard M
                        model.addConstr((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard K
                        model.addConstr((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard N
                        model.addConstr((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                    else:
                        raise Exception('Wrong!')


    matrix_commu_type = [[0, 0, 0],
              [Communication.ALL_GATHER.value, 0, Communication.ALL_TO_ALL.value], 
              [Communication.ALL_GATHER.value, Communication.ALL_TO_ALL.value, 0]];

    matrix_commu_size = [[0, 0, 0],
              [1, 0, 1], 
              [1, 1, 0]];


    matrix_commu_type = np.array(matrix_commu_type)
    matrix_commu_size = np.array(matrix_commu_size)

    edge_communication_type = model.addMVar((num_edge), name='edge_communication_type', vtype=gp.GRB.CONTINUOUS, lb=0)
    edge_communication_size = model.addMVar((num_edge), name='edge_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)
    for i in range(num_edge):
        model.addConstr(edge_communication_type[i] == upstream_sharding[i, :] @ matrix_commu_type @ downstream_sharding[i, :])
        model.addConstr(edge_communication_size[i] == upstream_sharding[i, :] @ matrix_commu_size @ downstream_sharding[i, :] * tensor_size[i])



    total_communication_size = model.addVar(name='total_communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)
    model.addConstr(total_communication_size == np.ones((num_kernel)) @ communication_size + np.ones((num_edge)) @ edge_communication_size)



    model.setObjective(total_communication_size, gp.GRB.MINIMIZE)
    model.optimize()




    # get variable values from gurobi program
    sharding = []
    communication_size = []
    communication_type = []
    edge_communication_size = []
    edge_communication_type = []

    for v in model.getVars():
        print(v.varName, v.x)

        if v.varName.startswith('sharding'):
            sharding.append(v.x)
        if v.varName.startswith('communication_size'):
            communication_size.append(v.x)
        if v.varName.startswith('communication_type'):
            communication_type.append(v.x)

        if v.varName.startswith('edge_communication_size'):
            edge_communication_size.append(v.x)
        if v.varName.startswith('edge_communication_type'):
            edge_communication_type.append(v.x)




       
        

    
    
    
# update kernels
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.WhichOneof('kernel_variant') == 'gemm_input1_weight':
        if sharding[i*5+0] == 1:
            kernel.gemm_input1_weight.sharding = Dim.OUTER.value+1
        elif sharding[i*5+1] == 1:
            kernel.gemm_input1_weight.sharding = Dim.M.value+1
        elif sharding[i*5+2] == 1:
            kernel.gemm_input1_weight.sharding = Dim.K.value+1
        elif sharding[i*5+3] == 1:
            kernel.gemm_input1_weight.sharding = Dim.N.value+1
        else:
            kernel.gemm_input1_weight.sharding = Dim.NO_DIM.value+1
         
        if kernel.gemm_input1_weight.communication_type != Communication.NO_COMMUNICATION.value:
            pass
        else:
            kernel.gemm_input1_weight.communication_size = float(communication_size[i])
            kernel.gemm_input1_weight.communication_type = int(communication_type[i])
        i += 1
        
    elif kernel.WhichOneof('kernel_variant') == 'gemm_input1_input2':
        if sharding[i*5+0] == 1:
            kernel.gemm_input1_input2.sharding = Dim.OUTER.value+1
        elif sharding[i*5+1] == 1:
            kernel.gemm_input1_input2.sharding = Dim.M.value+1
        elif sharding[i*5+2] == 1:
            kernel.gemm_input1_input2.sharding = Dim.K.value+1
        elif sharding[i*5+3] == 1:
            kernel.gemm_input1_input2.sharding = Dim.N.value+1
        else:
            kernel.gemm_input1_input2.sharding = Dim.NO_DIM.value+1
        
        if kernel.gemm_input1_input2.communication_type != Communication.NO_COMMUNICATION.value:
            pass
        else:
            kernel.gemm_input1_input2.communication_size = float(communication_size[i])
            kernel.gemm_input1_input2.communication_type = int(communication_type[i])
        i += 1
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1':
        if sharding[i*5+0] == 1:
            kernel.elementwise_input1.sharding = Dim.OUTER.value+1
        elif sharding[i*5+1] == 1:
            kernel.elementwise_input1.sharding = Dim.M.value+1
        elif sharding[i*5+2] == 1:
            kernel.elementwise_input1.sharding = Dim.K.value+1
        elif sharding[i*5+3] == 1:
            kernel.elementwise_input1.sharding = Dim.N.value+1
        else:
            kernel.elementwise_input1.sharding = Dim.NO_DIM.value+1
        
        if kernel.elementwise_input1.communication_type != Communication.NO_COMMUNICATION.value:
            pass
        else:        
            kernel.elementwise_input1.communication_size = float(communication_size[i])
            kernel.elementwise_input1.communication_type = int(communication_type[i])
        i += 1
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1_input2':
        if sharding[i*5+0] == 1:
            kernel.elementwise_input1_input2.sharding = Dim.OUTER.value+1
        elif sharding[i*5+1] == 1:
            kernel.elementwise_input1_input2.sharding = Dim.M.value+1
        elif sharding[i*5+2] == 1:
            kernel.elementwise_input1_input2.sharding = Dim.K.value+1
        elif sharding[i*5+3] == 1:
            kernel.elementwise_input1_input2.sharding = Dim.N.value+1
        else:
            kernel.elementwise_input1_input2.sharding = Dim.NO_DIM.value+1
            
        if kernel.elementwise_input1_input2.communication_type != Communication.NO_COMMUNICATION.value:
            pass
        else:
            kernel.elementwise_input1_input2.communication_size = float(communication_size[i])
            kernel.elementwise_input1_input2.communication_type = int(communication_type[i])
        i += 1
        
    else:
        raise Exception('Wrong!')


# update edges
i = 0
for connection in dse.dataflow_graph.connections:
    connection.communication_size = float(edge_communication_size[i])
    connection.communication_type = int(edge_communication_type[i])
    i += 1


# write to sharded binary
with open('./'+name+'/'+'dse_sharded.pb', "wb") as file:
    file.write(dse.SerializeToString())


# write to sharded text file
with open('./'+name+'/'+'dse_sharded.txt', "w") as file:
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


    graph.write_png('./'+name+'/'+'dataflow_graph_sharded.png') 
