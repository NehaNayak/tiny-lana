import sys
import random

allFile = open(sys.argv[1],'r')

lines = []

for line in allFile:
    lines.append(line)

listLen = len(lines)
indices = range(listLen)
random.shuffle(indices)

Test= [lines[i] for i in indices[:int(0.2*listLen)]]
Dev = [lines[i] for i in indices[int(0.2*listLen):int(0.3*listLen)]]
Train = [lines[i] for i in indices[int(0.3*listLen):]]

with open(sys.argv[1].replace(".txt","_Test.txt"),'w') as f:
    for line in Test:
        f.write(line)

with open(sys.argv[1].replace(".txt","_Dev.txt"),'w') as f:
    for line in Dev:
        f.write(line)

with open(sys.argv[1].replace(".txt","_Train.txt"),'w') as f:
    for line in Train:
        f.write(line)

