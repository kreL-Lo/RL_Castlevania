import numpy as np
import random
from collections import deque

d= deque()
d.append([[7,2,1],2,3,4])
d.append([[9,0,3],3,4,5])
d.append([[4,2,3],6,7,8])
d.append([[1,3,4],7,6,5])
f = np.array(random.sample(d,3))

g1 = np.array([x[0] for x in f])
g = np.array(f[:,0:1].tolist())
g = g[:,0]
print(g)
print(g1)
print(g[0].shape,g1[0].shape)

#print(g)
#print(g1)
