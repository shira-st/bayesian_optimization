#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from BayesianOptimization import BayesianOptimization


class ObjectiveFunction:
    def __init__(self, y):
        self.y = y
        
    def f(self, i):
        return self.y[i]
    

data_file = "data/esol.npy"
kernel_file = "kernel/esol_kernel.npy"

objective = ObjectiveFunction(np.load(data_file))
bo = BayesianOptimization(objective, np.load(kernel_file))
bo.optimize()

result = bo.result
plt.plot(list(range(len(result))), result)
plt.title("best observation")
plt.xlabel("iteration")
plt.ylabel("y")
plt.show()


# In[ ]:




