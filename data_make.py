import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import ipdb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import ArtistAnimation


class data_make():
    def baseline(self):
        x = np.linspace(-150, 150, 150)
        y = np.linspace(-150, 150, 150)
        X, Y = np.meshgrid(x, y)
        shape = X.shape
        z = np.c_[X.ravel(),Y.ravel()]
        x=y=0
        #ここでベースを強制的に0にしている
        Z=self.gaussian(z,x,y)*0
        Z=Z.reshape(shape)
        return X,Y,Z
    
    def gaussian(self,x,x_p,y_p):
        #2変数の分散共分散行列を指定
        # sigma=np.cov(x,y)
        sigma = np.array([[100,0],[0,100]])
        mu = np.array([x_p,y_p])
        # mu = np.array([1,1])
        #分散共分散行列の行列式
        det = np.linalg.det(sigma)
        # print(det)
        #分散共分散行列の逆行列
        inv = np.linalg.inv(sigma)
        n = x.ndim
        # print(inv)
        
        return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))
    
    def insert(self):
        #sin生成
        data_list=np.linspace(0,100,100)
        data_list_w=np.linspace(-100,100,100)
        data_sin_wabe=np.sin(2*np.pi*data_list)
        
        #meshgrit生成
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape
        
        z = np.c_[X.ravel(),Y.ravel()]
        print("Start insert")
        Z_v=self.gaussian(z,data_list_w[0]+np.random.randint(1,10)/10,data_sin_wabe[0]*50)
        Z_v = Z_v.reshape(shape)
        
        for i in range(len(data_list)-1):
        # for i in range(1):
            Z = self.gaussian(z,data_list_w[i+1]+np.random.randint(1,10)/10,data_sin_wabe[i+1]*50)
            Z = Z.reshape(shape)
            Z_v=np.dstack((Z_v,Z))
            # ipdb.set_trace()
            print("now calculation No.{}".format(i))
        return X,Y,Z_v



if __name__ == '__main__':
    '''
    定数
    '''
    dataset_num = 250
    sequence_length = 3
    t_start = -100.0
    sin_a = 2.0
    cos_a = 2.0
    sin_t = 25.0
    cos_t = 25.0
    calc_mode = "sin"
    # model pram
    input_size = 1
    output_size = 1
    hidden_size = 64
    batch_first = True
    # train pram
    lr = 0.001#学習率
    epochs = 15
    batch_size = 4
    test_size = 0.2
    
    dm=data_make()
    # lstm=PredictSimpleFormulaNet()
    X_b,Y_b,Z_b=dm.baseline()
    
    X,Y,Z=dm.insert()
    #--------
    #making graph
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ims=[]
    
    # frames = []  # 各フレームを構成する Artist 一覧
    # fig, ax = plt.subplots()
    
    # for i in range(99):
    #     fig = plt.figure()
    #     ax = Axes3D(fig)    
    #     print("now output No.{}".format(i))
    #     # Z[0,0,:]=0.002
    #     # Z[0,1,:]=-0.002
    #     ax.plot_wireframe(X, Y, Z[:,:,i])
    #     fig.savefig("save/img_{}.png".format(i))
    #     #1回matplotをリセットする必要がある
    #     #リセットしないと全てのグラフが残り続ける
    #     plt.clf()
    #     plt.close()
    
    count_time=0
    count_time1=0
    
    for i in range(499):
        print("a")
        if count_time==0:
            Z_stack=Z
            count_time+=1
            for l in range(len(Z)-1):
                print('debag',l)
                Z[:,:,l]=Z[:,:,l]+1
            # ipdb.set_trace()
            Z_stack=np.stack((Z_stack,Z))
        else:
            print(i)
            for l in range(len(Z)-1):
                print('debag',l)
                Z[:,:,l]=Z[:,:,l]+1
            # ipdb.set_trace()            
            Z_ex=np.expand_dims(Z,0)
            Z_stack=np.vstack((Z_stack,Z_ex))
        print("now doing ",i)
    print('fin')
    # Z_stack.tolist()
    np.save('np_save', Z_stack)
    ipdb.set_trace()
    
    # dict1 = {'Z_stack' : Z_stack}
    # file1 = open("dataset_for_vae-lstm.txt", "w") 
    # file1.write("%s" %(dict1))
    # file1.close()


# # フレームごとの Artist を作成する。
# for i in range(99):
#     print("now output No.{}".format(i))
#     # 折れ線グラフを作成する。
#     artists = ax.plot(Z[:,0,i])
#     # このフレームの Artist 一覧を追加する。
#     frames.append(artists)

# # アニメーションを作成する。
# ani = ArtistAnimation(fig, frames, interval=20)