#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from dataprovider.normalizer import MinMax01Scaler,MinMax11Scaler,StdScaler
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split


# In[2]:


class NCEPGetter(Dataset):
    def __init__(self, data, input_len):
        self.data = data
        self.input_len = input_len
        
    def __getitem__(self, index):
        chunk = self.data[index]
        return index, chunk[:self.input_len], chunk[self.input_len:]

    def __len__(self):
        return len(self.data)


# In[3]:


class NCEP:
    def __init__(self, config,):
        self.train_data_path = str(config['data']['train_data_path'])
        self.test_data_path = str(config['data']['test_data_path'])
        self.validate_ratio = float(config['data']['validate_ratio'])
        self.validate_random_state = int(config['data']['validate_random_state'])
        self.nomalize_method = str(config['data']['nomalizer'])
        self.input_len = int(config['model']['input_len'])
        self.output_len = int(config['model']['output_len'])
        self.batch_size = int(config['model']['batch_size'])
        self.test_batch_size = int(config['model']['test_batch_size'])
        self.chunk_interval = int(config['model']['chunk_interval'])
    
    def sequence_chunks(self, data):
        rows_count = len(data)
        chunk_len = self.input_len + self.output_len
        return np.array([range(i, i + chunk_len) for i in range(0, rows_count - chunk_len+1, self.chunk_interval)])        
    
    def train_loader(self,):
        
        self.train_vlidate_data = np.load(self.train_data_path)
        
        if self.nomalize_method == 'std':
            self.train_nomalizer = StdScaler(self.train_vlidate_data)
        elif self.nomalize_method == 'minmax01':
            self.train_nomalizer = MinMax01Scaler(self.train_vlidate_data)
        elif self.nomalize_method == 'minmax11':
            self.train_nomalizer = MinMax11Scaler(self.train_vlidate_data)
        else:
            raise Exception("Only std, minmax01, minmax11 nomalize methods are supported")
        
        self.train_vlidate_data = self.train_nomalizer.tranform()
        self.train_vlidate_index = self.sequence_chunks(self.train_vlidate_data)
        
        self.train_index, self.validate_index = train_test_split(self.train_vlidate_index,test_size=self.validate_ratio, random_state=self.validate_random_state, shuffle = True)
        
        self.train_getter = NCEPGetter(self.train_index, input_len= self.input_len)
        
        return DataLoader(dataset=self.train_getter,batch_size=self.batch_size,shuffle=True,)
    
    def validate_loader(self, ):
        self.validate_getter = NCEPGetter(self.validate_index, input_len= self.input_len)
        return DataLoader(dataset=self.validate_getter,batch_size=self.batch_size,shuffle=True,)
    
    def test_loader(self, ):
        
        self.test_data = np.load(self.test_data_path)
        self.test_data = np.transpose(self.test_data, (0, 3, 1, 2))
        
        p1, p2 = self.train_nomalizer.pivot_values()
        
        if self.nomalize_method == 'std':
            self.test_nomalizer = StdScaler(self.test_data, p1, p2)
        elif self.nomalize_method == 'minmax01':
            self.test_nomalizer = MinMax01Scaler(self.test_data, p1, p2)
        elif self.nomalize_method == 'minmax11':
            self.test_nomalizer = MinMax11Scaler(self.test_data, p1, p2)
        else:
            raise Exception("Only std, minmax01, minmax11 nomalize methods are supported")
        self.test_data = self.test_nomalizer.tranform()
        self.test_index = self.sequence_chunks(self.test_data)
        self.test_getter = NCEPGetter(self.test_index, input_len= self.input_len)
        return DataLoader(dataset=self.test_getter,batch_size=self.test_batch_size,shuffle=True,)


# In[6]:


if __name__ == '__main__':
    import configparser
    MODEL = 'lsntp'
    DATASET = 'ncep'
    config_file = '../{}_{}.config'.format(MODEL, DATASET)
    config = configparser.ConfigParser()
    config.read(config_file)
    ncep = NCEP(config)
    for (index,[batch_idx,data_index,target_index,]) in enumerate(ncep.train_loader()):
        print(batch_idx)
    for (index,[batch_idx,data_index,target_index,]) in enumerate(ncep.validate_loader()):
        print(batch_idx)
    for (index,[batch_idx,data_index,target_index,]) in enumerate(ncep.test_loader()):
        print(batch_idx)


# In[ ]:




