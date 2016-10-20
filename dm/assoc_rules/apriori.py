import numpy as np
from collections import OrderedDict

# read dataset and store it in a `special db`

dataset = open("../datasets/baskets.csv", 'r')

# parameters
special_db = {}
frequency_threshold = 3

for c, line in enumerate(dataset, 1):
    special_db[c] = line.replace('\n', '').split(',')

# reimplement this section :(
# here, get the frequent singletons

c = 0
items = OrderedDict()
counter = np.array([], dtype=int)

for basket in special_db.values():
    for item in basket:
        if item not in items:
            c += 1
            items[item] = c
            counter = np.append(counter, 0)
            counter[c-1] += 1
        else:
            pos = items[item]
            counter[pos - 1] += 1

# calculate the frequent doubletons

# recover the frequent singletons
new_indexes = np.array(counter) >= 3
items = np.array( items.keys() )

frequent_items = items[new_indexes]

print frequent_items
for basket in special_db.values():
    print basket
