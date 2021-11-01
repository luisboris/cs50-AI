import random

horses = {}
for horse in range(25):
    horses[horse] = random.random()

groups = {}
for i in range(5):
    group = {}
    for j in range(5):
        group[j+i*5] = horses[j+i*5]
    groups[i] = group


    

print(groups)