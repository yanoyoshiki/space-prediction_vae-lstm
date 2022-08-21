import numpy as np
import torch
import matplotlib.pyplot as plt
import ipdb

result=torch.load('first_experiment/ex_result').to('cpu').detach().numpy().copy()
index_num=np.arange(0,250)
# ipdb.set_trace()
plt.scatter(index_num,result)

plt.xlabel('sample index')
plt.ylabel('reconstraction error')
plt.legend()
plt.show()