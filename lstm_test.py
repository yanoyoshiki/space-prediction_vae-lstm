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


class PredictSimpleFormulaNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_first):#ハイパラ
        super(PredictSimpleFormulaNet, self).__init__()#class 継承の関数っぽい
        self.rnn = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            batch_first = batch_first)# definition LSTM
        self.output_layer = nn.Linear(hidden_size, output_size)#definition layer
        
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)


    def forward(self, inputs):
        h, _= self.rnn(inputs)
        output = self.output_layer(h[:, -1])
        
        return output



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
    fig = plt.figure()
    ax = Axes3D(fig)
    ims=[]
    
    # frames = []  # 各フレームを構成する Artist 一覧
    # fig, ax = plt.subplots()
    
    for i in range(99):
        fig = plt.figure()
        ax = Axes3D(fig)    
        print("now output No.{}".format(i))
        # Z[0,0,:]=0.002
        # Z[0,1,:]=-0.002
        ax.plot_wireframe(X, Y, Z[:,:,i])
        fig.savefig("save/img_{}.png".format(i))
        #1回matplotをリセットする必要がある
        #リセットしないと全てのグラフが残り続ける
        plt.clf()
        plt.close()
    
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
    ipdb.set_trace()
    
    dict1 = {'Z_stack' : Z_stack}
    file1 = open("dataset_for_vae-lstm.txt", "w") 
    file1.write("%s" %(dict1))
    file1.close()