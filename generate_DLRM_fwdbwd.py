import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--mlp_dim', type=int, required=True)
parser.add_argument('--bottom_num_mlp', type=int, required=True)
parser.add_argument('--top_num_mlp', type=int, required=True)
parser.add_argument('--pooled_row', type=int, required=True)
parser.add_argument('--num_table', type=int, required=True)
parser.add_argument('--emb', type=int, required=True)
parser.add_argument('--num_chip', type=int, required=True)
parser.add_argument('--micro_batch_size', type=int, required=True)
parser.add_argument('--word', type=int, required=True)
args = parser.parse_args()


mlp_dim = args.mlp_dim
bottom_num_mlp = args.bottom_num_mlp
top_num_mlp = args.top_num_mlp
pooled_row = args.pooled_row
num_table = args.num_table
emb = args.emb
num_chip = args.num_chip
micro_batch_size = args.micro_batch_size
word = args.word



memory_bytes = num_table * pooled_row * emb / num_chip * micro_batch_size * num_chip * word
network_bytes = num_table * pooled_row * emb / num_chip * micro_batch_size * word


num_mlp = bottom_num_mlp+top_num_mlp

cnt = 1
dataflow_graph = setup_pb2.Dataflow_Graph()

for i in range(1, num_mlp+1):
    kernel = dataflow_graph.kernels.add()
     
    kernel.name = "MLP_"+str(i)
    kernel.id = cnt
    kernel.config = -1
    kernel.fwd_bwd = 1
    kernel.type = 1
    kernel.gemm_input1_weight.outer = 1
    kernel.gemm_input1_weight.M = mlp_dim
    kernel.gemm_input1_weight.K = mlp_dim
    kernel.gemm_input1_weight.N = micro_batch_size
    kernel.gemm_input1_weight.input_tensor_size = mlp_dim*micro_batch_size*word
    kernel.gemm_input1_weight.weight_tensor_size = mlp_dim*mlp_dim*word
    kernel.gemm_input1_weight.output_tensor_size = mlp_dim*micro_batch_size*word
    kernel.gemm_input1_weight.tiling = 5
    
    if i == bottom_num_mlp:
        kernel.gemm_input1_weight.memory_size = memory_bytes
        kernel.gemm_input1_weight.communication_size = network_bytes
        kernel.gemm_input1_weight.communication_type = 2
    
    cnt += 1
   


for i in range(num_mlp, 0, -1):
    kernel = dataflow_graph.kernels.add()
    
    kernel.name = "MLP_"+str(i)+"_bwd"
    kernel.id = cnt
    kernel.config = -1
    kernel.fwd_bwd = 2
    kernel.type = 1
    kernel.gemm_input1_weight.outer = 1
    kernel.gemm_input1_weight.M = mlp_dim
    kernel.gemm_input1_weight.K = mlp_dim
    kernel.gemm_input1_weight.N = micro_batch_size
    kernel.gemm_input1_weight.input_tensor_size = mlp_dim*micro_batch_size*word
    kernel.gemm_input1_weight.weight_tensor_size = mlp_dim*mlp_dim*word
    kernel.gemm_input1_weight.output_tensor_size = mlp_dim*micro_batch_size*word
    kernel.gemm_input1_weight.tiling = 5
    
    if i == bottom_num_mlp:
        kernel.gemm_input1_weight.memory_size = memory_bytes
        kernel.gemm_input1_weight.communication_size = network_bytes
        kernel.gemm_input1_weight.communication_type = 2
        
    cnt += 1



for i in range(num_mlp, 0, -1):
    kernel = dataflow_graph.kernels.add()
    
    kernel.name = "MLP_"+str(i)+"_bwd_weight_update"
    kernel.id = cnt
    kernel.config = -1
    kernel.fwd_bwd = 2
    kernel.type = 1
    kernel.gemm_input1_input2.outer = 1
    kernel.gemm_input1_input2.M = mlp_dim
    kernel.gemm_input1_input2.K = micro_batch_size
    kernel.gemm_input1_input2.N = mlp_dim
    kernel.gemm_input1_input2.input_tensor_1_size = mlp_dim*micro_batch_size*word
    kernel.gemm_input1_input2.input_tensor_2_size = mlp_dim*micro_batch_size*word
    kernel.gemm_input1_input2.output_tensor_size = mlp_dim*mlp_dim*word
        
    kernel.gemm_input1_input2.communication_type = 4
    kernel.gemm_input1_input2.communication_size = mlp_dim*mlp_dim*word
    kernel.gemm_input1_input2.tiling = 5

    cnt += 1
    

cnt = 1
for i in range(1, num_mlp):
    connection = dataflow_graph.connections.add()

    connection.id = cnt
    connection.startIdx = i
    connection.endIdx = i+1
    
    cnt += 1
        
     
for i in range(num_mlp+1, 2*(num_mlp)):
    connection = dataflow_graph.connections.add()

    connection.id = cnt
    connection.startIdx = i
    connection.endIdx = i+1
    
    cnt += 1


for i in range(1, num_mlp):
    connection = dataflow_graph.connections.add()

    connection.id = cnt
    connection.startIdx = i
    connection.endIdx = 3*num_mlp - i
    
    cnt += 1



for i in range(num_mlp+1, 2*num_mlp):
    connection = dataflow_graph.connections.add()

    connection.id = cnt
    connection.startIdx = i
    connection.endIdx = 2*num_mlp + i-num_mlp+1
    
    cnt += 1
    
    
        
# write to text file
with open('./DLRM_fwdbwd.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)