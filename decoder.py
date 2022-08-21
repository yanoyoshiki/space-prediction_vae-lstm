from dis import dis
import os # tensorboardの出力先作成
import matplotlib.pyplot as plt # 可視化
import numpy as np # 計算
import torch # 機械学習フレームワークとしてpytorchを使用
import torch.nn as nn # クラス内で利用するモジュールのため簡略化
import torch.nn.functional as F # クラス内で利用するモジュールのため簡略化
from torch import optim # 最適化アルゴリズム
import ipdb
import pprint
from torchinfo import summary

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.z_dim=2
        self.x_dim = 10000
        
        self.dec_fc1 = nn.Linear(self.z_dim, 200) # デコーダ1層目
        self.dec_fc2 = nn.Linear(200, 400) # デコーダ2層目
        self.dec_drop = nn.Dropout(p=0.2) # 過学習を防ぐために最終層の直前にドロップアウト
        self.dec_fc3 = nn.Linear(400, self.x_dim) # デコーダ3層目
    
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

    def forward(self, x):
        y = self.decoder(x) # decoder部分
        return y


class lstm(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(lstm, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output_f, (hidden, cell) = self.rnn(inputs, hidden0)
        # ipdb.set_trace()
        output = self.output_layer(output_f[:, -1, :])
        # ipdb.set_trace()
        return output



if __name__ == '__main__':
    device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu") 
    
    raw_data=torch.load('first_experiment/z_from_encoder.pt') #潜在空間のデータがここに入ってくる
    
    #---------------------------------------------------------------------------------
    # lstm
    #---------------------------------------------------------------------------------
    
    lstm=lstm()
    PATH='first_experiment/lstm_model.pth'
    lstm.load_state_dict(torch.load(PATH), strict=False)
    lstm_model = lstm.to(device)
    
    #---------------------------------------------------------------------------------
    
    
    #---------------------------------------------------------------------------------
    # decoder
    #---------------------------------------------------------------------------------
    decoder = decoder()
    PATH='first_experiment/vae_model.pth'
    decoder.load_state_dict(torch.load(PATH), strict=False)
    decoder_model = decoder.to(device)
    
    #---------------------------------------------------------------------------------
    
    # use model
    x=lstm_model(raw_data)
    data=decoder_model(x)
    
    torch.save(data,'first_experiment/reconstraxtion_data')
    print("fin")
    
    ipdb.set_trace()