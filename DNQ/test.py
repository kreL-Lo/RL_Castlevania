import numpy as np
import random
from collections import deque
import random 
#random.seed(30)
f= [0.5,0.6,0.7]

EPSILON = 1
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.001
k =3
for i in range(k):
    if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)

print(EPSILON)
