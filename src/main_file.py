# Modules
import argparse, json
import label_personal_info
import anonymize_personal_info
import identification

parser = argparse.ArgumentParser(description='Program takes an input and output text file together with output formats')
parser.add_argument('--input', '-i', type=str, required=True)
parser.add_argument('--output', '-o', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    
    # Read input file, only text file
    with open(args.input) as file:
        data = file.read()
    
    output_data = identification.identify(data)
    
    # Save the output to the path in args.output
    if args.output:
        with open(args.output, 'w') as file:
            json.dump(output_data, file)
        #file = open(args.output, 'w')
        #file.write(output_data)
