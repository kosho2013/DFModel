import setup_pb2
import sys
import csv
import argparse
from google.protobuf import text_format
import yaml
import pandas as pd
import pydot
import pprint
import numpy as np

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='pls pass in the folder name under the current directory (named after the DL model)', required=True)
args = parser.parse_args()
name = args.name




# read in pd file
dse = setup_pb2.DSE()
with open('./'+name+'/'+'DSE.pb', "rb") as file:
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

for buffer in dse.dataflow_graph.buffers:    
    label = text_format.MessageToString(buffer)
    pydot_edge = pydot.Edge(dict[buffer.startIdx], dict[buffer.endIdx], label=label)
    graph.add_edge(pydot_edge)


graph.write_png('./'+name+'/'+name+'.png') 


