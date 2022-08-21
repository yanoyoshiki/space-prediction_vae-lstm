from faulthandler import disable
import numpy as np
import ipdb

counts_index=1
dist_x=np.load('np_save.npy')[:,:,:,:60]
ipdb.set_trace()
for l in range(250):
    for i in range(60):
        b=np.ravel(dist_x[l,:,:,i])
        if i == 0:
            a = b
        else:
            a = np.vstack([a,b])
    if counts_index==1:
        c = a
        counts_index=0
    else:
        c = np.vstack([c,a])
    print(c.shape)
print('fin')
np.save('for_vae_npdata/a=60---np_save_0-250', c)
ipdb.set_trace()



# counts_index=1
# dist_x=np.load('np_save.npy')[:,:,:,:60]
# for l in range(250):
#     for i in range(60):
#         b=np.ravel(dist_x[l+250,:,:,i])
#         if i == 0:
#             a = b
#         else:
#             a = np.vstack([a,b])
#     if counts_index==1:
#         c = a
#         counts_index=0
#     else:
#         c = np.vstack([c,a])
#     print(c.shape)
#     # ipdb.set_trace()
# print('fin')
# np.save('for_vae_npdata/a=60---np_save_251-500', c)


# fi=np.load('for_vae_npdata/np_save_251-500.npy')
# se=np.load('for_vae_npdata/np_save_0-250.npy')
# f_data=np.vstack([fi,se])
# np.save('for_vae_npdata/np_save_0-500', f_data)
# ipdb.set_trace()


#-------------------------------------------------------------
#for 3dim
#-------------------------------------------------------------

# dist_x=np.load('np_save.npy')
# for i in range(100):
#     for l in range(250):
#         b=np.ravel(dist_x[l,:,:,i])
#         if l == 0:
#             a = b
#         else:
#             a = np.vstack([a,b])
#     if i==0:
#         c=np.expand_dims(a,0) # (1,250,10000)
#     else:
#         a=np.expand_dims(a,0)
#         c=np.vstack([c,a])
#     print(c.shape)
# print('fin')
# np.save('3dim_npdata/np_save_0-250', c)


# dist_x=np.load('np_save.npy')
# for i in range(100):
#     for l in range(250):
#         b=np.ravel(dist_x[l+250,:,:,i])
#         if l == 0:
#             a = b
#         else:
#             a = np.vstack([a,b])
#     if i==0:
#         c=np.expand_dims(a,0)#(1,250,10000)
#     else:
#         a=np.expand_dims(a,0)
#         c=np.vstack([c,a])
#     print(c.shape)
# print('fin')
# np.save('3dim_npdata/np_save_251-500', c)


# fi=np.load('3dim_npdata/np_save_251-500.npy')
# se=np.load('3dim_npdata/np_save_0-250.npy')
# f_data=np.hstack([fi,se])
# np.save('3dim_npdata/np_save_0-500', f_data)
# ipdb.set_trace()
#-------------------------------------------------------------


#-------------------------------------------------------------
#for 2dim
#-------------------------------------------------------------
# dist_x=np.load('np_save.npy')
# dist_x=dist_x[:,:,:,:10]
# a=np.ravel(dist_x[0])
# for i in range(249):
#     b=np.ravel(dist_x[i+1])
#     a=np.vstack([a,b])
#     print(i)
# print('fin')
# # ipdb.set_trace()
# np.save('ver1_10/np_save_0-250', a)


# dist_x=np.load('np_save.npy')
# dist_x=dist_x[:,:,:,:10]
# a=np.ravel(dist_x[251])
# for i in range(249):
#     b=np.ravel(dist_x[i+251])
#     a=np.vstack([a,b])
#     print(i)
# print('fin')
# np.save('ver1_10/np_save_251-500', a)

# fi=np.load('ver1_10/np_save_251-500.npy')
# se=np.load('ver1_10/np_save_0-250.npy')
# f_data=np.vstack([fi,se])
# np.save('ver1_10/np_save_0-500', f_data)
# ipdb.set_trace()

#-------------------------------------------------------------
