import sys

for i, line in enumerate(sys.stdin):
    if i == 0:
        continue
        
    line = line.strip()
    id, sentence, label = line.strip().split('\t')
    if sentence:
        print(line)