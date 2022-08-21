from dis import dis
from mimetypes import init
import torch
import torch.nn.functional as F
import numpy as np
import ipdb

class kl():
    def torch_KL(self,first_dist,second_dist):
        torch_KL_d_value=F.kl_div(first_dist.log(), second_dist, None, None, 'sum')
        return torch_KL_d_value

    def trans(self,data):
        data_x=data
        for i in range(250):
            x = np.expand_dims(data_x[i*60:60*(i+1)],0)
            if i == 0:
                a = x
            else:
                a = np.vstack([a,x])
        return a
    
    def main(self):
        test_dist=torch.load('firts_experiment/reconstration_data')
        raw_dist=torch.tensor(np.load('for_vae_npdata/a=60---np_save_251-500.npy')).float()
        raw_dist=torch.tensor(self.trans(raw_dist)[:,-1,:])
        
        ipdb.set_trace()
        self.torch_KL(test_dist,raw_dist)
        
        for i in range(250):
            output=torch.unsqueeze(self.torch_KL(test_dist[i,:],raw_dist[i,:]),0)
            if i == 0:
                result_data=output
            else:
                result_data=torch.cat(result_data,output)
        return result_data
    
if __name__ == '__main__':
    result_data=kl().main()
    torch.save(result_data,'ex_result')
    ipdb.set_trace()