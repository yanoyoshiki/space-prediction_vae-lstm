from pyexpat import model
import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np
import ipdb

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

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

def mkDataSet(data_size, data_length=50, freq=60., noise=0.00):
    
    """
    params\n
    data_size : データセットサイズ\n
    data_length : 各データの時系列長\n
    freq : 周波数\n
    noise : ノイズの振幅\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    
    train_x = []
    train_t = []

    for offset in range(data_size):
        train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])
        
    return train_x, train_t

def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    
    return torch.tensor(batch_x), torch.tensor(batch_t)

def main():
    """
    自前のデータの際の条件
    inputdim=2
    outputdim=2
    
    hiddendim=??
    
    batchsize=これが一番何がイイかわからん
    
    """
    training_size = 10000
    test_size = 1000
    epochs_num = 1000
    hidden_size = 5
    batch_size = 100

    train_x, train_t = mkDataSet(training_size)
    test_x, test_t = mkDataSet(test_size)
    
    
    device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu") 
    
    #LSTMとして予測したい次元を示すinput_dim
    #もともとLSTM自体に入る次元は形が決まっていて
    model = Predictor(2, hidden_size, 2).to(device)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()
            
            data, label = mkRandomBatch(train_x, train_t, batch_size)
            # ipdb.set_trace()
            # data=torch.cat((data,data),2)
            
            #------------------------------------------------------------------
            #chanege input data that is from vae.py(z)
            #------------------------------------------------------------------
            # z=torch.load('z_from_vae.pt')
            z=torch.tensor(np.load('z/first-tolstm.npy')).to(device)
            data, label = z[:,:58,:],z[:,59,:]
            # ipdb.set_trace()
            #------------------------------------------------------------------
            #dataの構造として[データ種,データ長,入力次元数]という構造になっている
            #ここで1サンプルのデータを入力に使用したい場合はdata[0,:,:]で次元拡張を行えばモデルを使用できる
            # ipdb.set_trace()
            output = model(data)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            # ipdb.set_trace()
            training_accuracy += np.sum(np.abs((output.data - label.data).to('cpu').numpy()) < 0.1)
        
        #test
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size]).to(device), torch.tensor(test_t[offset:offset+batch_size]).to(device)
            data=torch.cat((data,data),2)
            ipdb.set_trace()
            output = model(data, None)
            
            test_accuracy += np.sum(np.abs((output.data - label.data).to('cpu').numpy()) < 0.1)
        
        training_accuracy /= training_size
        test_accuracy /= test_size
        
        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))
        #--------------------------------------------------------------------------------------------------
        #model save
        #--------------------------------------------------------------------------------------------------
    model_path = 'lstm-model.pth'
    torch.save(model.state_dict(), model_path)




def model_ex_using():
    print('asd')
    
if __name__ == '__main__':
    main()