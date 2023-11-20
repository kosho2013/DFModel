import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--seq', type=int, required=True)
parser.add_argument('--num_head', type=int, required=True)
parser.add_argument('--head_dim', type=int, required=True)
args = parser.parse_args()


hidden = args.hidden
seq = args.seq
num_head = args.num_head
head_dim = args.head_dim


if num_head * head_dim != hidden:
    raise Exception('Wrong!')


word = 2



dataflow_graph = setup_pb2.Dataflow_Graph()

for i in range(1, 19):
    kernel = dataflow_graph.kernels.add()
    
    if i == 1:
        kernel.name = 'Add_Prev_Layer'
        kernel.id = i
        kernel.config = -1
        
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 2:
        kernel.name = 'LayerNorm_1'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 3:
        kernel.name = 'Q'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = head_dim
        kernel.batch_gemm_elementwise_outer_m_k_n.K = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = hidden*hidden*word
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 4:
        kernel.name = 'K'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = head_dim
        kernel.batch_gemm_elementwise_outer_m_k_n.K = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = hidden*hidden*word
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 5:
        kernel.name = 'V'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = head_dim
        kernel.batch_gemm_elementwise_outer_m_k_n.K = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1.0
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = hidden*hidden*word
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
    
    elif i == 6:
        kernel.name = 'MHA_GEMM_1'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.K = head_dim
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = seq*seq*num_head*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 3
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 4
        
    elif i == 7:
        kernel.name = 'SOFTMAX'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = seq*seq*num_head*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = seq*seq*num_head*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 6
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 8:
        kernel.name = 'DropOut_1'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = seq*seq*num_head*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = seq*seq*num_head*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 7
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 9:
        kernel.name = 'MHA_GEMM_2'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = head_dim
        kernel.batch_gemm_elementwise_outer_m_k_n.K = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = seq*seq*num_head*word
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 5
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 8

    elif i == 10:
        kernel.name = 'PROJ_GEMM'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = hidden*hidden*word
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 9
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 11:
        kernel.name = 'DropOut_2'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 10
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 12:
        kernel.name = 'Add_1'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 11
        
    elif i == 13:
        kernel.name = "LayerNorm_2"
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 12
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1

    elif i == 14:
        kernel.name = "FFN0"
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = 4*hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = 4*hidden*hidden*word
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = 4*hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 13
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 15:
        kernel.name = 'GeLU'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = 4*hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = 4*hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = 4*hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 14
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
    elif i == 16:
        kernel.name = 'FFN1'
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 4*hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = 4*hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = 4*hidden*hidden*word
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 15
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
          
    elif i == 17:
        kernel.name = "DropOut_3"
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 16
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
    
    elif i == 18:
        kernel.name = 'Add_2'
        kernel.id =  18
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.M = hidden
        kernel.batch_gemm_elementwise_outer_m_k_n.K = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 2
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 12
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 17
          
          

# write to text file
with open('./generated_LLM.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)
    