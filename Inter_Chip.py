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
for kernel in dse.dataflow_graph.kernels:
    kernel_name.append(kernel.name)


output_tensor_size = []
for kernel in dse.dataflow_graph.kernels:
    output_tensor_size.append(kernel.batch_gemm_elementwise_outer_m_k_m.output_tensor_size)


kernel_type = []
for kernel in dse.dataflow_graph.kernels:
    kernel_type.append(kernel.batch_gemm_elementwise_outer_m_k_m.type)


outer = []
for kernel in dse.dataflow_graph.kernels:
    outer.append(kernel.batch_gemm_elementwise_outer_m_k_m.outer)


startIdx = []
endIdx = []
for connection in dse.dataflow_graph.connections:
    startIdx.append(connection.startIdx)
    endIdx.append(connection.endIdx)

node_dict = {}
i = 0
for kernel in dse.dataflow_graph.kernels:
    node_dict[kernel.id] = i
    i += 1


edge_dict = {}
i = 0
for connection in dse.dataflow_graph.connections:
    edge_dict[(connection.startIdx, connection.endIdx)] = i
    i += 1
num_edge = len(edge_dict)



input_tensor_1_id = []
input_tensor_2_id = []
for kernel in dse.dataflow_graph.kernels:
    input_tensor_1_id.append(kernel.batch_gemm_elementwise_outer_m_k_m.input_tensor_1_id)
    input_tensor_2_id.append(kernel.batch_gemm_elementwise_outer_m_k_m.input_tensor_2_id)



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
            print(id, indegree[id])
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
    connection.tensor_size = output_tensor_size[node_dict[connection.startIdx]]








class Communication(Enum):
    NO_COMMUNICATION = 0
    ALL_REDUCE = 1
    ALL_TO_ALL = 2
    ALL_GATHER = 3


model = gp.Model()
model.params.NonConvex = 2
model.Params.Threads = 10
model.params.MIPGap = 0.1    # 10%
model.params.TimeLimit = 300  # 5 minutes


sharding = model.addMVar((num_kernel, 4), name='sharding', vtype=gp.GRB.BINARY)
communication_type = model.addMVar((num_kernel), name='communication_type', vtype=gp.GRB.INTEGER, lb=Communication.NO_COMMUNICATION.value, ub=Communication.ALL_GATHER.value)
communication_size = model.addMVar((num_kernel), name='communication_size', vtype=gp.GRB.CONTINUOUS, lb=0)

for i in range(num_kernel):
    model.addConstr(sharding[i, 3] == 0)
    model.addConstr(np.ones((4)) @ sharding[i, :] == 1)

    if outer[i] > 1:
        model.addConstr(sharding[i, 0] == 1)

        # if K is sharded
        model.addConstr((sharding[i, 2] == 1) >> (communication_type[i] == Communication.ALL_REDUCE.value))
        model.addConstr((sharding[i, 2] == 1) >> (communication_size[i] == output_tensor_size[i]))

        # if K is not sharded
        model.addConstr((sharding[i, 2] == 0) >> (communication_type[i] == Communication.NO_COMMUNICATION.value))
        model.addConstr((sharding[i, 2] == 0) >> (communication_size[i] == 0))
    else:
        model.addConstr(sharding[i, 0] == 0)

        # if K is sharded
        model.addConstr((sharding[i, 2] == 1) >> (communication_type[i] == Communication.ALL_REDUCE.value))
        model.addConstr((sharding[i, 2] == 1) >> (communication_size[i] == output_tensor_size[i]))

        # if K is not sharded
        model.addConstr((sharding[i, 2] == 0) >> (communication_type[i] == Communication.NO_COMMUNICATION.value))
        model.addConstr((sharding[i, 2] == 0) >> (communication_size[i] == 0))


# RR, RS, SR
output_tensor = model.addMVar((num_kernel, 3), name='output_tensor', vtype=gp.GRB.BINARY) # (outer,M),N
for i in range(num_kernel):
    model.addConstr(np.ones((3)) @ output_tensor[i, :] == 1)

    model.addConstr((sharding[i, 0] == 1) >> (output_tensor[i, 2] == 1)) # shard outer
    model.addConstr((sharding[i, 1] == 1) >> (output_tensor[i, 2] == 1)) # shard M
    model.addConstr((sharding[i, 2] == 1) >> (output_tensor[i, 0] == 1)) # shard K
    model.addConstr((sharding[i, 3] == 1) >> (output_tensor[i, 1] == 1)) # shard N


# RR, RS, SR
input_tensor_1 = model.addMVar((num_kernel, 3), name='input_tensor_1', vtype=gp.GRB.BINARY) # outer,K,N
for i in range(num_kernel):
    model.addConstr(np.ones((3)) @ input_tensor_1[i, :] == 1)

    model.addConstr((sharding[i, 0] == 1) >> (input_tensor_1[i, 2] == 1)) # shard outer
    model.addConstr((sharding[i, 1] == 1) >> (input_tensor_1[i, 0] == 1)) # shard M
    model.addConstr((sharding[i, 2] == 1) >> (input_tensor_1[i, 2] == 1)) # shard K
    model.addConstr((sharding[i, 3] == 1) >> (input_tensor_1[i, 1] == 1)) # shard N



# RR, RS, SR
input_tensor_2 = model.addMVar((num_kernel, 3), name='input_tensor_2', vtype=gp.GRB.BINARY) # outer,M,K
for i in range(num_kernel):
    model.addConstr(np.ones((3)) @ input_tensor_2[i, :] == 1)

    model.addConstr((sharding[i, 0] == 1) >> (input_tensor_2[i, 2] == 1)) # shard outer
    model.addConstr((sharding[i, 1] == 1) >> (input_tensor_2[i, 2] == 1)) # shard M
    model.addConstr((sharding[i, 2] == 1) >> (input_tensor_2[i, 1] == 1)) # shard K
    model.addConstr((sharding[i, 3] == 1) >> (input_tensor_2[i, 0] == 1)) # shard N




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
    startNodeId = startIdx[i]
    endNodeId = endIdx[i]

    if input_tensor_1_id[node_dict[endNodeId]] == startNodeId:
        model.addConstr(edge_communication_type[i] == output_tensor[node_dict[startNodeId], :] @ matrix_commu_type @ input_tensor_1[node_dict[endNodeId], :])
        model.addConstr(edge_communication_size[i] == output_tensor[node_dict[startNodeId], :] @ matrix_commu_size @ input_tensor_1[node_dict[endNodeId], :] * output_tensor_size[node_dict[startNodeId]])
    else:
        model.addConstr(edge_communication_type[i] == output_tensor[node_dict[startNodeId], :] @ matrix_commu_type @ input_tensor_2[node_dict[endNodeId], :])
        model.addConstr(edge_communication_size[i] == output_tensor[node_dict[startNodeId], :] @ matrix_commu_size @ input_tensor_2[node_dict[endNodeId], :] * output_tensor_size[node_dict[startNodeId]])





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
    if sharding[i*4+0] == 1:
        kernel.batch_gemm_elementwise_outer_m_k_m.sharding = 1
    elif sharding[i*4+1] == 1:
        kernel.batch_gemm_elementwise_outer_m_k_m.sharding = 2
    elif sharding[i*4+2] == 1:
        kernel.batch_gemm_elementwise_outer_m_k_m.sharding = 3
    else:
        kernel.batch_gemm_elementwise_outer_m_k_m.sharding = 4
    
    kernel.batch_gemm_elementwise_outer_m_k_m.communication_size = float(communication_size[i])
    kernel.batch_gemm_elementwise_outer_m_k_m.communication_type = int(communication_type[i])
    i += 1



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
