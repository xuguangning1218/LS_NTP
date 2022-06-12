#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model.network import Network
import torch
from torch import nn,optim
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from tqdm import tqdm
# In[2]:


class Model:
    
    def __init__(self, config):
        self.num_gpus = int(config['model']['num_gpus'])
        self.learning_rate = float(config['model']['learning_rate'])
        self.device = torch.device(config['model']['device'])
        self.model_save_path = str(config['model']['model_save_path'])
        self.resume_checkpoint = str(config['model']['resume_checkpoint'])
        self.height = int(config['data']['height'])
        self.width = int(config['data']['width'])
        self.input_len = int(config['model']['input_len'])
        self.output_len = int(config['model']['output_len'])
        self.lsntp_downsample = True if str(config['model']['lsntp_downsample']) == 'True' else False
        self.config = config
        
    def get_model(self):
        if self.num_gpus == 1:
            self.network = Network(self.config)
        else:
            self.network = nn.DataParallel(Network(self.config)) 
        self.network = self.network.to(self.device) 
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        return self.network
            
    def train(self, data, input_index, target_index):
        input_data = torch.Tensor(data[input_index],).float().to(self.device)
        target_data = torch.Tensor(data[target_index],).float().to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        output_data = self.network(input_data, )
        loss = self.criterion(output_data, target_data)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validate(self,data, input_index, output_index):
        input_data = torch.Tensor(data[input_index], ).float().to(self.device)
        target_data = torch.Tensor(data[output_index],).float().to(self.device)
        output_data = self.network(input_data, )
        val_loss = self.criterion(output_data, target_data)
        return val_loss.item()
    
        
    def test(self,data, dataloader, nomalizer):
        mae = np.zeros((self.output_len, self.height, self.width))
        mse = np.zeros((self.output_len, self.height, self.width))
        ssim = np.zeros((self.output_len))
        psnr = np.zeros((self.output_len))
        counter = 0
        
        with torch.no_grad():
            for batch_idx,input_index,target_index, in tqdm(dataloader) :
                input_data = torch.Tensor(data[input_index]).float().to(self.device)
                target = data[target_index]
                pred = self.network(input_data, )
                pred = pred.cpu().detach().numpy()
                pred_reverse = nomalizer.reverse(pred)
                target_reverse = nomalizer.reverse(target)
                for _b in range(target.shape[0]):
                    for _t in range(target.shape[1]): 
                        for _c in range(target.shape[2]):
                            mae[_t] += np.abs(target_reverse[_b, _t, _c].astype(np.float32) - pred_reverse[_b, _t, _c])
                            mse[_t] += (target_reverse[_b, _t, _c].astype(np.float32) - pred_reverse[_b, _t, _c])**2
                            ssim[_t] += structural_similarity(target_reverse[_b, _t, _c, :, :].astype(np.float32), pred_reverse[_b, _t, _c, :, :])
                            psnr[_t] += peak_signal_noise_ratio(target_reverse[_b, _t, _c, :, :].astype(np.float32), pred_reverse[_b, _t, _c, :, :], data_range=255)
                counter += target.shape[0]
        return mae, mse, ssim, psnr, counter
    
    def save(self,epoch, model_name=None):
        
        if model_name is None:
            model_name = 'checkpoint-'+str(epoch)+'.pth'
        
        checkpoint = {
         'epoch': epoch+1, # next epoch
         'model': self.network.state_dict(),
         'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.model_save_path + model_name)
        print("save ", model_name, " to ", self.model_save_path, " successfully")
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # for ncep model loading
    # Since the data volume is small in ncep, we only save the model state
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#     def load(self, model_name=None):
#         start_epoch = 70
#         if model_name is None:
#             model_name = self.resume_checkpoint
#         checkpoint = torch.load(self.model_save_path + model_name)
#         self.network.load_state_dict(checkpoint)
#         print("loaded model successfully")
#         return start_epoch
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # for era5 model loading
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def load(self, model_name=None):
        if model_name is None:
            model_name = self.resume_checkpoint
        if self.lsntp_downsample:
        # for ear5 
            checkpoint = torch.load(self.model_save_path + model_name)
            start_epoch = checkpoint['epoch']
            self.network.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded model successfully")
            return start_epoch
        else:
        # for ncep
        # Since the data volume is small in ncep, we only save the model state    
            start_epoch = 70
            checkpoint = torch.load(self.model_save_path + model_name)
            self.network.load_state_dict(checkpoint)
            print("loaded model successfully")
            return start_epoch

    def param_counter(self,):
        num_params = 0
        for param in self.network.parameters():
            num_params += param.numel()
        print('Number of params: %.2fM' % (num_params / 1e6))
    
# In[ ]:




