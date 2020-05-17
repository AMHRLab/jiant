import jsonlines
import csv



with open("train-labels.lst") as l:
    labels = l.readlines()
    labels = [int(int(line.rstrip()) == 2) for line in labels]

with jsonlines.open('dev.jsonl') as f:
    instances = []
    for line in f.iter():
        instances.append(str(line)) # or whatever else you'd like to do


with open('preds.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    preds = []
    for row in csv_reader:
        preds.append(int(row[0]))
        line_count += 1
    print(f'Processed {line_count} lines.')


output = open("output.txt", 'a')
for i in range(0, len(preds)):
    if preds[i] != labels[i]:
        output.write("Label: " + str(labels[i]) + " | " + instances[i] + "\n")

l.close()
f.close()
output.close()