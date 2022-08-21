from faulthandler import disable
import numpy as np
import ipdb


# dist_x=np.load('np_save.npy')
# a=np.ravel(dist_x[0])
# for i in range(250):
#     b=np.ravel(dist_x[i+1])
#     a=np.vstack([a,b])
#     print(i)
# print('fin')
# np.save('np_save_0-250', a)


# dist_x=np.load('np_save.npy')
# a=np.ravel(dist_x[251])
# for i in range(249):
#     b=np.ravel(dist_x[i+251])
#     a=np.vstack([a,b])
#     print(i)
# print('fin')
# np.save('np_save_251-500', a)


fi=np.load('np_save_251-500.npy')
se=np.load('np_save_0-250.npy')
ipdb.set_trace()
f_data=np.vstack([fi,se])
np.save('np_save_0-500', f_data)