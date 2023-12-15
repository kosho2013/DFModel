import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot
import math



# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, required=True)
parser.add_argument('--B', type=int, required=True)
parser.add_argument('--X', type=int, required=True)
parser.add_argument('--Y', type=int, required=True)
parser.add_argument('--word', type=int, required=True)
args = parser.parse_args()


N = args.N
B = args.B
X = args.X
Y = args.Y
word = args.word


        #Y
    ###########
#X  ###########
    ###########
    

iterations = int(N/B)    
    
dataflow_graph = setup_pb2.Dataflow_Graph()

for i in range(iterations):
    side = N - i*B
    P = math.ceil(side / X)
    Q = math.ceil(side / Y)
    
    
    kernel = dataflow_graph.kernels.add()
    kernel.name = "Iteration_"+str(i+1)+"_FACT"
    kernel.id = i*4+1
    kernel.config = i
    kernel.type = 1
    kernel.gemm_input1_weight.outer = 1
    kernel.gemm_input1_weight.M = P
    kernel.gemm_input1_weight.K = B
    kernel.gemm_input1_weight.N = B
    kernel.gemm_input1_weight.input_tensor_size = B*B*word
    kernel.gemm_input1_weight.weight_tensor_size = P*B*word
    kernel.gemm_input1_weight.output_tensor_size = P*B*word
    kernel.gemm_input1_weight.tiling = 5
    
    
    kernel = dataflow_graph.kernels.add()
    kernel.name = "Iteration_"+str(i+1)+"_BCAST"
    kernel.id = i*4+2
    kernel.config = i
    kernel.type = 1
    kernel.gemm_input1_weight.outer = 1
    kernel.gemm_input1_weight.M = 1
    kernel.gemm_input1_weight.K = 1
    kernel.gemm_input1_weight.N = 1
    kernel.gemm_input1_weight.input_tensor_size = word
    kernel.gemm_input1_weight.weight_tensor_size = word
    kernel.gemm_input1_weight.output_tensor_size = word
    kernel.gemm_input1_weight.tiling = 5
    
    if Y == 1:
        kernel.gemm_input1_weight.communication_size = 0
    else:
        kernel.gemm_input1_weight.communication_type = 6
        kernel.gemm_input1_weight.communication_size = P*B*word
    
    
    kernel = dataflow_graph.kernels.add()
    kernel.name = "Iteration_"+str(i+1)+"_SWAP"
    kernel.id = i*4+3
    kernel.config = i
    kernel.type = 1
    kernel.gemm_input1_weight.outer = 1
    kernel.gemm_input1_weight.M = 1
    kernel.gemm_input1_weight.K = 1
    kernel.gemm_input1_weight.N = 1
    kernel.gemm_input1_weight.input_tensor_size = word
    kernel.gemm_input1_weight.weight_tensor_size = word
    kernel.gemm_input1_weight.output_tensor_size = word
    kernel.gemm_input1_weight.tiling = 5
    
    if X == 1:
        kernel.gemm_input1_weight.communication_size = 0
    else:
        kernel.gemm_input1_weight.communication_type = 5
        kernel.gemm_input1_weight.communication_size = Q*B*word
    
    
    
    kernel = dataflow_graph.kernels.add()
    kernel.name = "Iteration_"+str(i+1)+"_UPDATE"
    kernel.id = i*4+4
    kernel.config = i
    kernel.type = 1
    kernel.gemm_input1_weight.outer = 1
    kernel.gemm_input1_weight.M = P
    kernel.gemm_input1_weight.K = B
    kernel.gemm_input1_weight.N = Q
    kernel.gemm_input1_weight.input_tensor_size = B*Q*word
    kernel.gemm_input1_weight.weight_tensor_size = P*B*word
    kernel.gemm_input1_weight.output_tensor_size = P*Q*word
    kernel.gemm_input1_weight.tiling = 5
    
    
cnt = 1
for i in range(iterations):
    connection = dataflow_graph.connections.add()
    connection.id = cnt
    connection.startIdx = i*4+1
    connection.endIdx = i*4+2
    cnt += 1
    
    connection = dataflow_graph.connections.add()
    connection.id = cnt
    connection.startIdx = i*4+1
    connection.endIdx = i*4+3
    cnt += 1
    
    connection = dataflow_graph.connections.add()
    connection.id = cnt
    connection.startIdx = i*4+1
    connection.endIdx = i*4+4
    cnt += 1
    

for i in range(iterations-1):
    connection = dataflow_graph.connections.add()
    connection.id = cnt
    connection.startIdx = i*4+2
    connection.endIdx = (i+1)*4+1
    cnt += 1

    
    connection = dataflow_graph.connections.add()
    connection.id = cnt
    connection.startIdx = i*4+3
    connection.endIdx = (i+1)*4+1
    cnt += 1
    
    
    connection = dataflow_graph.connections.add()
    connection.id = cnt
    connection.startIdx = i*4+4
    connection.endIdx = (i+1)*4+1
    cnt += 1
    
        
# write to text file
with open('./HPL.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)