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

for i in range(1, 36):
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = -1
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 17
    

    elif i == 19:
        kernel.name = "Loss_bwd"
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
        
    elif i == 20:
        kernel.name = "DropOut_3_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 19
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
    
    elif i == 21:
        kernel.name = "FFN1_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 20
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
    
    elif i == 22:
        kernel.name = "GeLU_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 21
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
      
    elif i == 23:
        kernel.name = "FFN0_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 23
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1

    elif i == 24:
        kernel.name = "LayerNorm_2_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 24
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1

    elif i == 25:
        kernel.name = "DropOut_2_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 26
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
      
    elif i == 26:
        kernel.name = "PROJ_GEMM_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 27
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
      
    elif i == 27:
        kernel.name = "MHA_GEMM_2_bwd1"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 28
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 5

    elif i == 28:
        kernel.name = "MHA_GEMM_2_bwd2"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 28
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 8
      
    elif i == 29:
        kernel.name = "V_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 31
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
      
    elif i == 30:
        kernel.name = "DropOut_1_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 30
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
      
    elif i == 31:
        kernel.name = "SOFTMAX_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 34
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
      
    elif i == 32:
        kernel.name = "MHA_GEMM_1_bwd1"
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = head_dim
        kernel.batch_gemm_elementwise_outer_m_k_n.K = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = seq*seq*num_head*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 35
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 4
      
    elif i == 33:
        kernel.name = "MHA_GEMM_1_bwd2"
        kernel.id = i
        kernel.config = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.outer = num_head
        kernel.batch_gemm_elementwise_outer_m_k_n.M = head_dim
        kernel.batch_gemm_elementwise_outer_m_k_n.K = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.N = seq
        kernel.batch_gemm_elementwise_outer_m_k_n.type = 1
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_size = seq*seq*num_head*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.weight_tensor_size = -1
        kernel.batch_gemm_elementwise_outer_m_k_n.output_tensor_size = hidden*seq*word
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 35
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = 3
      
    elif i == 34:
        kernel.name = "Q_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 36
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
      
    elif i == 35:
        kernel.name = "K_bwd"
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
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_1_id = 37
        kernel.batch_gemm_elementwise_outer_m_k_n.input_tensor_2_id = -1
        
          


for i in range(1, 42):
    connection = dataflow_graph.connections.add()
    
    if i == 1:
        connection.id = i
        connection.startIdx = 1
        connection.endIdx = 2
    
    elif i == 2:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 3
    
    elif i == 3:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 4
    
    elif i == 4:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 5
    
    elif i == 5:
        connection.id = i
        connection.startIdx = 3
        connection.endIdx = 6
    
    elif i == 6:
        connection.id = i
        connection.startIdx = 4
        connection.endIdx = 6

    elif i == 7:
        connection.id = i
        connection.startIdx = 6
        connection.endIdx = 7
    
    elif i == 8:
        connection.id = i
        connection.startIdx = 7
        connection.endIdx = 8
    
    elif i == 9:
        connection.id = i
        connection.startIdx = 5
        connection.endIdx = 9
        
    elif i == 10:
        connection.id = i
        connection.startIdx = 8
        connection.endIdx = 9
    
    elif i == 11:
        connection.id = i
        connection.startIdx = 9
        connection.endIdx = 10
    
    elif i == 12:
        connection.id = i
        connection.startIdx = 10
        connection.endIdx = 11
    
    elif i == 13:
        connection.id = i
        connection.startIdx = 11
        connection.endIdx = 12
    
    elif i == 14:
        connection.id = i
        connection.startIdx = 1
        connection.endIdx = 12
    
    elif i == 15:
        connection.id = i
        connection.startIdx = 12
        connection.endIdx = 13
    
    elif i == 16:
        connection.id = i
        connection.startIdx = 13
        connection.endIdx = 14
    
    elif i == 17:
        connection.id = i
        connection.startIdx = 14
        connection.endIdx = 15
    
    elif i == 18:
        connection.id = i
        connection.startIdx = 15
        connection.endIdx = 16
        
    elif i == 19:
        connection.id = i
        connection.startIdx = 16
        connection.endIdx = 17
        
    elif i == 20:
        connection.id = i
        connection.startIdx = 17
        connection.endIdx = 18
    
    elif i == 21:
        connection.id = i
        connection.startIdx = 12
        connection.endIdx = 18
        
        
        
        
    elif i == 22:
        connection.id = i
        connection.startIdx = 19
        connection.endIdx = 20
    
    elif i == 23:
        connection.id = i
        connection.startIdx = 20
        connection.endIdx = 21

    elif i == 24:
        connection.id = i
        connection.startIdx = 21
        connection.endIdx = 22

    elif i == 25:
        connection.id = i
        connection.startIdx = 22
        connection.endIdx = 23

    elif i == 26:
        connection.id = i
        connection.startIdx = 23
        connection.endIdx = 24

    elif i == 27:
        connection.id = i
        connection.startIdx = 24
        connection.endIdx = 25

    elif i == 28:
        connection.id = i
        connection.startIdx = 25
        connection.endIdx = 26

    elif i == 29:
        connection.id = i
        connection.startIdx = 26
        connection.endIdx = 27

    elif i == 30:
        connection.id = i
        connection.startIdx = 5
        connection.endIdx = 27

    elif i == 31:
        connection.id = i
        connection.startIdx = 26
        connection.endIdx = 28

    elif i == 32:
        connection.id = i
        connection.startIdx = 8
        connection.endIdx = 28

    elif i == 33:
        connection.id = i
        connection.startIdx = 27
        connection.endIdx = 30
   
    elif i == 34:
        connection.id = i
        connection.startIdx = 28
        connection.endIdx = 29

    elif i == 35:
        connection.id = i
        connection.startIdx = 30
        connection.endIdx = 31

    elif i == 36:
        connection.id = i
        connection.startIdx = 31
        connection.endIdx = 32

    elif i == 37:
        connection.id = i
        connection.startIdx = 31
        connection.endIdx = 33

    elif i == 38:
        connection.id = i
        connection.startIdx = 4
        connection.endIdx = 32

    elif i == 39:
        connection.id = i
        connection.startIdx = 3
        connection.endIdx = 33

    elif i == 40:
        connection.id = i
        connection.startIdx = 32
        connection.endIdx = 34

    elif i == 41:
        connection.id = i
        connection.startIdx = 33
        connection.endIdx = 35 
        
        
        
# write to text file
with open('./fwdbwd.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)