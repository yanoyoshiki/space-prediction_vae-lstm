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



class VAE_LSTM(nn.Module):
    def __init__(self, z_dim,hidden_size,batch_first):
        """コンストラクタ

        Args:
            z_dim (int): 潜在空間の次元数

        Returns:
            None.

        Note:
            eps (float): オーバーフローとアンダーフローを防ぐための微小量
            """
        
        super(VAE_LSTM, self).__init__() # VAEクラスはnn.Moduleを継承しているため親クラスのコンストラクタを呼ぶ必要がある
        self.eps = np.spacing(1) # オーバーフローとアンダーフローを防ぐための微小量 printすると2.2ぐらいになる
        # self.x_dim = 28 * 28 # MNISTの場合は28×28の画像であるため
        # self.x_dim = 1000000
        self.x_dim = 500
        #今回の場合3(軸)*6(軌道のサンプリング方法)とか
        self.z_dim = z_dim # インスタンス化の際に潜在空間の次元数は自由に設定できる
        #----------------------------------------------
        self.enc_fc1 = nn.Linear(self.x_dim, 400) # エンコーダ1層目 全結合層で28^2の次元から400にしてる
        self.enc_fc2 = nn.Linear(400, 200) # エンコーダ2層目
        #----------------------------------------------
        self.enc_fc3_mean = nn.Linear(200, z_dim) # 近似事後分布の平均
        self.enc_fc3_logvar = nn.Linear(200, z_dim) # 近似事後分布の分散の対数
        #----------------------------------------------
        #LSTM層
        input_size=1
        
        self.rnn = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            batch_first = batch_first)# definition LSTM
        #----------------------------------------------
        
        #----------------------------------------------
        self.dec_fc1 = nn.Linear(z_dim, 200) # デコーダ1層目
        self.dec_fc2 = nn.Linear(200, 400) # デコーダ2層目
        self.dec_drop = nn.Dropout(p=0.2) # 過学習を防ぐために最終層の直前にドロップアウト
        self.dec_fc3 = nn.Linear(400, self.x_dim) # デコーダ3層目



        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

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

    def decoder(self, z):
        """デコーダ

        Args:
            z (torch.tensor): (バッチサイズ, z_dim)サイズの潜在変数

        Returns:
            y (torch.tensor): (バッチサイズ, 入力次元数)サイズの再構成データ
        """
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))


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
        ipdb.set_trace()
        z = torch.unsqueeze(z,2)
        ipdb.set_trace()
        h, _= self.rnn(z)
        ipdb.set_trace()
        y = self.decoder(h) # decoder部分
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var)) # KLダイバージェンス計算
        reconstruction = torch.sum(x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)) # 再構成誤差計算
        return [KL, reconstruction], z, y




if __name__ == '__main__':
    batch_size = 10
    batch_first = True
    hidden_size = 64
    
    V=VAE_LSTM(3,hidden_size,batch_first)
    
    # ipdb.set_trace()
    # GPUが使える場合はGPU上で動かす
    device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu") 
    # VAEクラスのコンストラクタに潜在変数の次元数を渡す
    model = VAE_LSTM(2,hidden_size,batch_first).to(device)
    
    # 今回はoptimizerとしてAdamを利用
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 最大更新回数は1000回
    num_epochs = 1000
    # 検証データのロスとその最小値を保持するための変数を十分大きな値で初期化しておく
    loss_valid = 10 ** 7
    loss_valid_min = 10 ** 7
    # early stoppingを判断するためのカウンタ変数
    num_no_improved = 0
    # tensorboardに記録するためのカウンタ変数
    num_batch_train = 0
    num_batch_valid = 0
    
    dist_x=np.load('np_save_0-500.npy')
    counts_index=0
    # 学習開始-------------------------------
    for num_iter in range(num_epochs):
        model.train() # 学習前は忘れずにtrainモードにしておく
        for x, t in dataloader_train: # dataloaderから訓練データを抽出する
            x=x[:,0:500]
            # x=dist_x
            # ipdb.set_trace()        
            x=x.to(device)
            ipdb.set_trace()
            lower_bound, _, _ = model(x, device) # VAEにデータを流し込む 引数として取り出すのは[KL,再構築誤差]
            loss = -sum(lower_bound) # lossは負の下限
            model.zero_grad() # 訓練時のpytorchのお作法
            loss.backward()
            optimizer.step()
            num_batch_train += 1
            counts_index += 1
            print(counts_index)
        counts_index=0
        num_batch_train -= 1 # 次回のエポックでつじつまを合わせるための調整
    #----------------------------------------

        # 検証開始
        model.eval() # 検証前は忘れずにevalモードにしておく
        loss = []
        for x, t in dataloader_valid: # dataloaderから検証データを抽出する
            x=x[:,0:500]
            # x=dist_x
            x=x.to(device)
            # ipdb.set_trace()
            lower_bound, _, _ = model(x, device) # VAEにデータを流し込む
            loss.append(-sum(lower_bound).cpu().detach().numpy())
            num_batch_valid += 1
        num_batch_valid -= 1 # 次回のエポックでつじつまを合わせるための調整
        loss_valid = np.mean(loss)
        loss_valid_min = np.minimum(loss_valid_min, loss_valid)
        print(f"[EPOCH{num_iter + 1}] loss_valid: {int(loss_valid)} | Loss_valid_min: {int(loss_valid_min)}")

        # もし今までのlossの最小値よりも今回のイテレーションのlossが大きければカウンタ変数をインクリメントする
        if loss_valid_min < loss_valid:
            num_no_improved += 1
            print(f"{num_no_improved}回連続でValidationが悪化しました")
        # もし今までのlossの最小値よりも今回のイテレーションのlossが同じか小さければカウンタ変数をリセットする
        else:
            num_no_improved = 0
            torch.save(model.state_dict(), f"./z_{model.z_dim}.pth")
        # カウンタ変数が10回に到達したらearly stopping
        if (num_no_improved >= 10):
            print(f"{num_no_improved}回連続でValidationが悪化したため学習を止めます")
            break
ipdb.set_trace()
#---------------------------------------------------
