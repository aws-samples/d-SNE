import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

out_list = []
with open(args.input) as f:
    for i, line in enumerate(f):
        path, label = line.split()
        out_list.append([i, int(label), path])

with open(args.output, 'w') as f:
    csv_writer = csv.writer(f, delimiter='\t')
    for line in out_list:
        csv_writer.writerow(line)
