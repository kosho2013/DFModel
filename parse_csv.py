import setup_pb2
import csv
import argparse
from google.protobuf import text_format


# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='pls pass in the folder name under the current directory (named after the DL model), in that folder, there should be five csv files: 1. csv file for kernels called kernels.csv; 2. csv file for connections called connections.csv; 3. csv file for the architctural parameters of a single accelerator called accelerator.csv; 4. csv file for the entire multi-chip system topology called system.csv; 5. csv file for off-chip memory and network unit cost called cost.csv.', required=True)
args = parser.parse_args()
name = args.name



dse = setup_pb2.DSE()

# read kernels
with open('./'+name+'/kernels.csv', mode ='r') as file:
    lines = list(csv.reader(file))

    for i in range(1, len(lines)):
        kernel = dse.dataflow_graph.kernels.add()
        
        fields = lines[i][0].split(';')
        
        kernel.name = str(fields[0])
        kernel.id = int(fields[1])
        kernel.outer = int(fields[2])
        kernel.M = int(fields[3])
        kernel.K = int(fields[4])
        kernel.N = int(fields[5])
        kernel.type = int(fields[6])   
        kernel.input_tensor_1_size = float(fields[7])   
        kernel.input_tensor_2_size = float(fields[8])
        kernel.weight_tensor_size = float(fields[9])   
        kernel.output_tensor_size = float(fields[10])
        kernel.input_tensor_1_id = int(fields[11])
        kernel.input_tensor_2_id = int(fields[12])

    


                

# read connections
with open('./'+name+'/connections.csv', mode ='r') as file:
    lines = list(csv.reader(file))

    for i in range(1, len(lines)):
        connection = dse.dataflow_graph.connections.add()
        
        fields = lines[i][0].split(';')
        
        connection.startIdx = int(fields[0])
        connection.endIdx = int(fields[1])

        


# read accelerator
with open('./'+name+'/accelerator.csv', mode ='r') as file:
    lines = list(csv.reader(file))
    
    fields = lines[1][0].split(';')
    
    dse.system.accelerator.name = str(fields[0])
    dse.system.accelerator.core = int(fields[1])
    dse.system.accelerator.systolic_width = int(fields[2])
    dse.system.accelerator.systolic_height = int(fields[3])  
    dse.system.accelerator.sram_cap = float(fields[4])  
    dse.system.accelerator.freq = float(fields[5])  
    dse.system.accelerator.dram_bw = float(fields[6])  


# read system
with open('./'+name+'/system.csv', mode ='r') as file:
    lines = list(csv.reader(file))
    
    fields = lines[1][0].split(';')
    
    dse.system.name = str(fields[0])
    dse.system.topo = int(fields[1])
    dse.system.num_chip = int(fields[2])
    dse.system.link_bw_x = float(fields[3])  
    dse.system.link_bw_y = float(fields[4])  
    dse.system.link_bw_z = float(fields[5])


# read cost
with open('./'+name+'/cost.csv', mode ='r') as file:
    lines = list(csv.reader(file))
    
    fields = lines[1][0].split(';')
    
    dse.cost.link_unit_price = float(fields[0])
    dse.cost.switch_unit_price = float(fields[1])
    dse.cost.dram_unit_price = float(fields[2])  





# write to binary
with open('./'+name+'/'+'dse.pb', "wb") as file:
    file.write(dse.SerializeToString())


# write to text file
with open('./'+name+'/'+'dse.txt', "w") as file:
    text_format.PrintMessage(dse, file)






# read in pd file
dse = setup_pb2.DSE()
with open('./'+name+'/'+'dse.pb', "rb") as file:
    dse.ParseFromString(file.read())