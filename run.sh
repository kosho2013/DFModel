protoc --python_out=. setup.proto
python parse_input.py --name=$1
python inter_chip.py --name=$1
python kernel_level.py --name=$1

