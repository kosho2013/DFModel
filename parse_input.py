import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='pls pass in the folder name under the current directory (named after the DL model), in that folder, there should be five csv files: 1. csv file for kernels called kernels.csv; 2. csv file for connections called connections.csv; 3. csv file for the architctural parameters of a single accelerator called accelerator.csv; 4. csv file for the entire multi-chip system topology called topology.csv; 5. csv file for off-chip memory and network unit cost called cost.csv.', required=True)
args = parser.parse_args()
name = args.name




# read setup.txt
dse = setup_pb2.DSE()
with open('./'+name+'/user_input/'+'setup.txt', "r") as file:
    text = file.read()
    text_format.Parse(text, dse)



# write to binary
with open('./'+name+'/'+'dse.pb', "wb") as file:
    file.write(dse.SerializeToString())


# write to text file
with open('./'+name+'/'+'dse.txt', "w") as file:
    text_format.PrintMessage(dse, file)





# read in binary file
dse = setup_pb2.DSE()
with open('./'+name+'/'+'dse.pb', "rb") as file:
    dse.ParseFromString(file.read())


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


graph.write_png('./'+name+'/'+'dataflow_graph.png') 
