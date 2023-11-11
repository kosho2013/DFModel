import dataflow_graph_pb2
import sys
import csv
import argparse
from google.protobuf import text_format


# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='pls pass in the folder name under the current directory (we prefer the folder is name after the neurla network model name), in that folder, there should be two csv files, one csv file for kernels called kernels.csv and another one for buffers called buffers.csv', required=True)
# parser.add_argument('--batch_size', type=str, help='pls pass in the micro-batch size of the neural network model', required=True)
# parser.add_argument('--num_layer', type=str, help='pls pass in the number of layers in the neural network model', required=True)

args = parser.parse_args()
name = args.name
# batch_size = args.batch_size
# num_layer = args.num_layer


# create graph
dataflow_graph = dataflow_graph_pb2.Dataflow_Graph()

# read kernels
with open('./'+name+'/kernels.csv', mode ='r') as file:
    lines = list(csv.reader(file))

    for i in range(1, len(lines)):
        kernel = dataflow_graph.kernels.add()
        
        fields = lines[i][0].split(';')
        
        kernel.name = str(fields[0])
        kernel.id = int(fields[1])
        kernel.outer = int(fields[2])
        kernel.M = int(fields[3])
        kernel.K = int(fields[4])
        kernel.N = int(fields[5])
        kernel.type = int(fields[6])   
        
# read buffers
with open('./'+name+'/buffers.csv', mode ='r') as file:
    lines = list(csv.reader(file))

    for i in range(1, len(lines)):
        buffer = dataflow_graph.buffers.add()
        
        fields = lines[i][0].split(';')
        
        buffer.name = str(fields[0])
        buffer.id = int(fields[1])
        buffer.tensor_size = int(fields[2])
        buffer.startIdx = int(fields[3])
        buffer.endIdx = int(fields[4])
        buffer.type = int(fields[5])       


# write to binary
with open('./'+name+'/'+name+'.pb', "wb") as file:
    file.write(dataflow_graph.SerializeToString())


# write to text file
with open('./'+name+'/'+name+'.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)
        
        
