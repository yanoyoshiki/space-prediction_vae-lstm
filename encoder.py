from dis import dis
import os # tensorboardの出力先作成
import matplotlib.pyplot as plt # 可視化
import numpy as np # 計算
import torch # 機械学習フレームワークとしてpytorchを使用
import torch.nn as nn # クラス内で利用するモジュールのため簡略化
import torch.nn.functional as F # クラス内で利用するモジュールのため簡略化
from torch import optim # 最適化アルゴリズム
from torchvision import datasets, transforms # データセットの準備
import ipdb
import pprint
from torchinfo import summary

# MNISTのデータをとってくるときに一次元化する前処理
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# trainデータとtestデータに分けてデータセットを取得
dataset_train_valid = datasets.MNIST("./", train=True, download=True, transform=transform)
dataset_test = datasets.MNIST("./", train=False, download=True, transform=transform)

# trainデータの20%はvalidationデータとして利用
size_train_valid = len(dataset_train_valid) # 60000
size_train = int(size_train_valid * 0.8) # 48000
size_valid = size_train_valid - size_train # 12000
dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train_valid, [size_train, size_valid])

# 取得したデータセットをDataLoader化する
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1000, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1000, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1000, shuffle=False)



class encoder_z(nn.Module):
    def __init__(self):
        
        
        """コンストラクタ

        Args:
            z_dim (int): 潜在空間の次元数

        Returns:
            None.

        Note:
            eps (float): オーバーフローとアンダーフローを防ぐための微小量
            """
        super(encoder_z, self).__init__() # VAEクラスはnn.Moduleを継承しているため親クラスのコンストラクタを呼ぶ必要がある
        self.eps = np.spacing(1) # オーバーフローとアンダーフローを防ぐための微小量 printすると2.2ぐらいになる
        
        # self.x_dim = 28 * 28 # MNISTの場合は28×28の画像であるため
        self.x_dim = 10000
        z_dim=2
        
        #今回の場合3(軸)*6(軌道のサンプリング方法)とか
        self.z_dim = z_dim # インスタンス化の際に潜在空間の次元数は自由に設定できる
        #----------------------------------------------
        self.enc_fc1 = nn.Linear(self.x_dim, 400) # エンコーダ1層目 全結合層で28^2の次元から400にしてる
        self.enc_fc2 = nn.Linear(400, 200) # エンコーダ2層目
        #----------------------------------------------
        self.enc_fc3_mean = nn.Linear(200, z_dim) # 近似事後分布の平均
        self.enc_fc3_logvar = nn.Linear(200, z_dim) # 近似事後分布の分散の対数

#----------------------------------------------------

    def encoder(self, x):
        """エンコーダ

        Args:
            x (torch.tensor): (バッチサイズ, 入力次元数)サイズの入力データ

        Returns:
            mean (torch.tensor): 近似事後分布の平均
            logvar (torch.tensor): 近似事後分布の分散の対数
        """
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)



    def sample_z(self, mean, log_var, device):
        """Reparametrization trickに基づく潜在変数Zの疑似的なサンプリング

        Args:
            mean (torch.tensor): 近似事後分布の平均
            logvar (torch.tensor): 近似事後分布の分散の対数
            device (String): GPUが使える場合は"cuda"でそれ以外は"cpu"

        Returns:
            z (torch.tensor): (バッチサイズ, z_dim)サイズの潜在変数
        """
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5 * log_var)


    def forward(self, x, device):
        """順伝播処理

        Args:
            x (torch.tensor): (バッチサイズ, 入力次元数)サイズの入力データ
            device (String): GPUが使える場合は"cuda"でそれ以外は"cpu"

        Returns:
            KL (torch.float): KLダイバージェンス
            reconstruction (torch.float): 再構成誤差
            z (torch.tensor): (バッチサイズ, z_dim)サイズの潜在変数
            y (torch.tensor): (バッチサイズ, 入力次元数)サイズの再構成データ            
        """
        mean, log_var = self.encoder(x.to(device)) # encoder部分
        z = self.sample_z(mean, log_var, device) # Reparametrization trick部分
        return  z 

class data_tran():
    def transf(self,z):
        #z=(15000,2)
        for i in range(250):
            ex_data=torch.unsqueeze(z[i*60:(i+1)*60:,:],0)
            if i==0:
                stack_data=ex_data
            else:
                stack_data=torch.cat((stack_data,ex_data),0)
                print(stack_data.shape)
        return stack_data



if __name__ == '__main__':
    device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu") 
    encoder_z=encoder_z()
    PATH='vae-model.pth'
    encoder_z.load_state_dict(torch.load(PATH), strict=False)
    model = encoder_z.to(device)
    
    #must be inputed 251-250 npdata  
    x=torch.tensor(np.load('for_vae_npdata/a=60---np_save_251-500.npy')*(1/100)).float()
    # x=torch.tensor(np.load('for_vae_npdata/a=60---np_save_0-250.npy')*(1/100)).float()
    
    
    z=model(x,device)
    
    #--------------------------
    # transform data shape for input to lstm model
    #--------------------------
    z_trans=data_tran().transf(z)
    
    
    torch.save(z_trans,'first_experiment/z_from_encoder')
    # print('fin')
    ipdb.set_trace()