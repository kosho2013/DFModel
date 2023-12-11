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

for i in range(1, 42):
    kernel = dataflow_graph.kernels.add()
    
    if i == 1:
        kernel.name = "Add_Prev_Layer"
        kernel.id = 1
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq		
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4
        
    elif i == 2:
        kernel.name = "LayerNorm_1"
        kernel.id = 2
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1  
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq  
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4
          
    elif i == 3:
        kernel.name = "Q"
        kernel.id = 3
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = num_head
        kernel.gemm_input1_weight.M = head_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 4:
        kernel.name = "K"
        kernel.id = 4
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = num_head
        kernel.gemm_input1_weight.M = head_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 5:
        kernel.name = "V"
        kernel.id = 5
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = num_head
        kernel.gemm_input1_weight.M = head_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 6:
        kernel.name = "MHA_GEMM_1"
        kernel.id = 6
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = seq
        kernel.gemm_input1_input2.K = head_dim
        kernel.gemm_input1_input2.N = seq
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = seq*seq*num_head*word
        kernel.gemm_input1_input2.tiling = 4
        
    elif i == 7:
        kernel.name = "SOFTMAX"
        kernel.id = 7
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = seq
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.output_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.tiling = 4

    elif i == 8:
        kernel.name = "DropOut_1"
        kernel.id = 8
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = seq
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.output_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.tiling = 4

    elif i == 9:
        kernel.name = "MHA_GEMM_2"
        kernel.id = 9
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = head_dim
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = seq
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = seq*seq*num_head*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_input2.tiling = 4

    elif i == 10:
        kernel.name = "PROJ_GEMM"
        kernel.id = 10
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = hidden
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4       

    elif i == 11:
        kernel.name = "DropOut_2"
        kernel.id = 11
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 12:
        kernel.name = "Add_1"
        kernel.id = 12
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1_input2.outer = 1
        kernel.elementwise_input1_input2.M = hidden
        kernel.elementwise_input1_input2.N = seq  
        kernel.elementwise_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.elementwise_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.elementwise_input1_input2.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1_input2.tiling = 4

    elif i == 13:
        kernel.name = "LayerNorm_2"
        kernel.id = 13
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1  
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 14:
        kernel.name = "FFN0"
        kernel.id = 14
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = 4*hidden
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = 4*hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = 4*hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4
    
    elif i == 15:
        kernel.name = "GeLU"
        kernel.id = 15
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = 4*hidden
        kernel.elementwise_input1.N = seq 
        kernel.elementwise_input1.input_tensor_size = 4*hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = 4*hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 16:
        kernel.name = "FFN1"
        kernel.id = 16
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1 
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = hidden
        kernel.gemm_input1_weight.K = 4*hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = 4*hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = 4*hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 17:
        kernel.name = "DropOut_3"
        kernel.id = 17
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 18:
        kernel.name = "Add_2"
        kernel.id = 18
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1_input2.outer = 1
        kernel.elementwise_input1_input2.M = hidden
        kernel.elementwise_input1_input2.N = seq
        kernel.elementwise_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.elementwise_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.elementwise_input1_input2.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1_input2.tiling = 4

    elif i == 19:
        kernel.name = "Loss_bwd"
        kernel.id = 19
        kernel.fwd_bwd = 2
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4
    
    elif i == 20:
        kernel.name = "DropOut_3_bwd"
        kernel.id = 20
        kernel.type = 2
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 21:
        kernel.name = "FFN1_bwd"
        kernel.id = 21
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = 4*hidden
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = 4*hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = 4*hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 22:
        kernel.name = "GeLU_bwd"
        kernel.id = 22
        kernel.type = 2
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = 4*hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = 4*hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = 4*hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 23:
        kernel.name = "FFN0_bwd"
        kernel.id = 23
        kernel.type = 1
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = hidden
        kernel.gemm_input1_weight.K = 4*hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = 4*hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = 4*hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4        

    elif i == 24:
        kernel.name = "LayerNorm_2_bwd"
        kernel.id = 24
        kernel.type = 2
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 25:
        kernel.name = "DropOut_2_bwd"
        kernel.id = 25
        kernel.fwd_bwd = 2
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 26:
        kernel.name = "PROJ_GEMM_bwd"
        kernel.id = 26
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = hidden
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 27:
        kernel.name = "MHA_GEMM_2_bwd1"
        kernel.id = 27
        kernel.type = 1
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = seq
        kernel.gemm_input1_input2.K = head_dim
        kernel.gemm_input1_input2.N = seq  
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = seq*seq*num_head*word
        kernel.gemm_input1_input2.tiling = 4

    elif i == 28:
        kernel.name = "MHA_GEMM_2_bwd2"
        kernel.id = 28
        kernel.type = 1
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = head_dim
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = seq
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = seq*seq*num_head*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_input2.tiling = 4
        
    elif i == 29:
        kernel.name = "V_bwd"
        kernel.id = 29
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = hidden
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4
    
    elif i == 30:
        kernel.name = "DropOut_1_bwd"
        kernel.id = 30
        kernel.type = 2
        kernel.fwd_bwd = 2
        kernel.config = -1  
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = seq
        kernel.elementwise_input1.N = seq   
        kernel.elementwise_input1.input_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.output_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.tiling = 4

    elif i == 31:
        kernel.name = "SOFTMAX_bwd"
        kernel.id = 31
        kernel.fwd_bwd = 2
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = seq
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.output_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.tiling = 4

    elif i == 32:
        kernel.name = "MHA_GEMM_1_bwd1"
        kernel.id = 32
        kernel.type = 1
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = head_dim
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = seq
        kernel.gemm_input1_input2.input_tensor_1_size = seq*seq*num_head*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_input2.tiling = 4

    elif i == 33:
        kernel.name = "MHA_GEMM_1_bwd2"
        kernel.id = 33
        kernel.type = 1
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = head_dim
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = seq
        kernel.gemm_input1_input2.input_tensor_1_size = seq*seq*num_head*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_input2.tiling = 4

    elif i == 34:
        kernel.name = "Q_bwd"
        kernel.id = 34
        kernel.type = 1
        kernel.fwd_bwd = 2
        kernel.config = -1
        kernel.gemm_input1_weight.outer = num_head
        kernel.gemm_input1_weight.M = head_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq 
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 35:
        kernel.name = "K_bwd"
        kernel.id = 35
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = num_head
        kernel.gemm_input1_weight.M = head_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4
   
    elif i == 36:
        kernel.name = "FFN1_bwd_weight_update"
        kernel.id = 36
        kernel.config = -1
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.gemm_input1_input2.outer = 1
        kernel.gemm_input1_input2.M = 4*hidden
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = hidden
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = 4*hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = 4*hidden*hidden*word
        kernel.gemm_input1_input2.tiling = 3
        kernel.gemm_input1_input2.communication_type = 4
        kernel.gemm_input1_input2.communication_size = 4*hidden*hidden*word

    elif i == 37:
        kernel.name = "FFN0_bwd_weight_update"
        kernel.id = 37
        kernel.config = -1
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.gemm_input1_input2.outer = 1
        kernel.gemm_input1_input2.M = hidden
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = 4*hidden
        kernel.gemm_input1_input2.input_tensor_1_size = 4*hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = 4*hidden*hidden*word
        kernel.gemm_input1_input2.tiling = 3
        kernel.gemm_input1_input2.communication_type = 4
        kernel.gemm_input1_input2.communication_size = 4*hidden*hidden*word
        
    elif i == 38:
        kernel.name = "PROJ_GEMM_bwd_weight_update"
        kernel.id = 38
        kernel.config = -1
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.gemm_input1_input2.outer = 1
        kernel.gemm_input1_input2.M = hidden
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = hidden
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*hidden*word
        kernel.gemm_input1_input2.tiling = 3
        kernel.gemm_input1_input2.communication_type = 4
        kernel.gemm_input1_input2.communication_size = hidden*hidden*word

    elif i == 39:
        kernel.name = "V_bwd_weight_update"
        kernel.id = 39
        kernel.config = -1
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.gemm_input1_input2.outer = 1
        kernel.gemm_input1_input2.M = hidden
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = hidden
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*hidden*word
        kernel.gemm_input1_input2.tiling = 3
        kernel.gemm_input1_input2.communication_type = 4
        kernel.gemm_input1_input2.communication_size = hidden*hidden*word

    elif i == 40:
        kernel.name = "K_bwd_weight_update"
        kernel.id = 40
        kernel.config = -1
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.gemm_input1_input2.outer = 1
        kernel.gemm_input1_input2.M = hidden
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = hidden
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*hidden*word
        kernel.gemm_input1_input2.tiling = 3
        kernel.gemm_input1_input2.communication_type = 4
        kernel.gemm_input1_input2.communication_size = hidden*hidden*word
        
    elif i == 41:
        kernel.name = "Q_bwd_weight_update"
        kernel.id = 41
        kernel.config = -1
        kernel.fwd_bwd = 2
        kernel.type = 1
        kernel.gemm_input1_input2.outer = 1
        kernel.gemm_input1_input2.M = hidden
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = hidden
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*hidden*word
        kernel.gemm_input1_input2.tiling = 3
        kernel.gemm_input1_input2.communication_type = 4
        kernel.gemm_input1_input2.communication_size = hidden*hidden*word


for i in range(1, 54):
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
    
    elif i == 42:
        connection.id = i
        connection.startIdx = 20
        connection.endIdx = 36

    elif i == 43:
        connection.id = i
        connection.startIdx = 15
        connection.endIdx = 36
    
    elif i == 44:
        connection.id = i
        connection.startIdx = 22
        connection.endIdx = 37
    
    elif i == 45:
        connection.id = i
        connection.startIdx = 13
        connection.endIdx = 37

    elif i == 46:
        connection.id = i
        connection.startIdx = 25
        connection.endIdx = 38

    elif i == 47:
        connection.id = i
        connection.startIdx = 9
        connection.endIdx = 38

    elif i == 48:
        connection.id = i
        connection.startIdx = 28
        connection.endIdx = 39
    
    elif i == 49:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 39
    
    elif i == 50:
        connection.id = i
        connection.startIdx = 33
        connection.endIdx = 40
    
    elif i == 51:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 40
    
    elif i == 52:
        connection.id = i
        connection.startIdx = 32
        connection.endIdx = 41
    
    elif i == 53:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 41
        
        
# write to text file
with open('./LLM_fwdbwd.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)