#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
from model.lsconvlstm import LSConvLSTM
# In[2]:


class Encoder(nn.Module):
    def __init__(self, config, node_num):
        
        self.in_channel = int(config['model']['in_channel'])
        self.kernel_size = int(config['model']['kernel_size'])
        self.conv_dim = int(config['model']['conv_dim'])
        self.lsntp_downsample = True if str(config['model']['lsntp_downsample']) == 'True' else False
        print('self.lsntp_downsample:', self.lsntp_downsample, config['model']['lsntp_downsample'])
        super(Encoder,self).__init__()
        if self.lsntp_downsample:
            self.conv2d = nn.Conv2d(in_channels=self.in_channel, 
                 out_channels=self.in_channel, 
                 kernel_size=self.kernel_size ,
                 stride=2,
                 padding=0)
        self.convrnn1 = LSConvLSTM(self.in_channel, self.conv_dim, (self.kernel_size, self.kernel_size), node_num)
        self.convrnn2 = LSConvLSTM(self.conv_dim, self.conv_dim, (self.kernel_size, self.kernel_size), node_num)

    def forward(self, x, node_embeddings):
        state_list = []
        
        if self.lsntp_downsample:
            x_list = []
            for t in range(x.size(1)):
                x_list.append(self.conv2d(x[:, t, :, :, :]))
            x = torch.stack(x_list, dim=1)
        
        encoder1, state = self.convrnn1(x, node_embeddings)
        state_list.append(state)
        encoder2, state = self.convrnn2(encoder1, node_embeddings)
        state_list.append(state)
        return state_list


# In[3]:


class Decoder(nn.Module):
    def __init__(self, config, node_num):
        
        self.in_channel = int(config['model']['in_channel'])
        self.kernel_size = int(config['model']['kernel_size'])
        self.conv_dim = int(config['model']['conv_dim'])
        self.lsntp_downsample = True if str(config['model']['lsntp_downsample']) == 'True' else False
        
        super(Decoder, self).__init__()
        
        self.convrnn3 = LSConvLSTM(self.conv_dim, self.conv_dim, (self.kernel_size, self.kernel_size), node_num)
        self.convrnn4 = LSConvLSTM(self.conv_dim, self.conv_dim, (self.kernel_size, self.kernel_size), node_num)
        
        if self.lsntp_downsample:
            self.conv2dtranspose = nn.ConvTranspose2d(
                 in_channels=self.conv_dim, 
                 out_channels=self.in_channel, 
                 kernel_size=self.kernel_size,
                 stride=2,
                 padding=0)
        else:
            self.conv2d = nn.Conv2d(
                 in_channels=self.conv_dim, 
                 out_channels=self.in_channel, 
                 kernel_size=self.kernel_size,
                 stride=1,
                 padding=1)
        
    def forward(self, x, state, node_embeddings):
        decoder1, _ = self.convrnn3(x, node_embeddings, state[0])
        decoder2, _ = self.convrnn4(decoder1, node_embeddings, state[1])
        
        if self.lsntp_downsample:
            output = []
            for t in range(x.size(1)):
                output.append(self.conv2dtranspose(decoder2[:, t, :, :, :]))
        else:
            output = []
            for t in range(x.size(1)):
                output.append(self.conv2d(decoder2[:, t, :, :, :]))
            
        return torch.stack(output ,dim = 1)


# In[4]:

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.node_embedding_dim = int(config['model']['node_embedding_dim'])
        self.height = int(config['data']['height'])
        self.width = int(config['data']['width'])
        self.lsntp_downsample = True if str(config['model']['lsntp_downsample']) == 'True' else False
        if self.lsntp_downsample:
            self.node_num = (self.height//2)*(self.width//2)
        else:
            self.node_num = (self.height)*(self.width)
        self.encoder = Encoder(config, self.node_num)
        self.decoder = Decoder(config, self.node_num)
        self.input_len = int(config['model']['input_len'])
        # adaptive graph construction
        self.node_embeddings = nn.Parameter(torch.Tensor(self.node_num, self.node_embedding_dim), requires_grad=True)
        nn.init.xavier_normal_(self.node_embeddings)
    def forward(self, x):
        states = self.encoder(x, self.node_embeddings)
        decoder_input = torch.zeros((states[0][0].size(0), self.input_len, states[0][0].size(1),states[0][0].size(2),states[0][0].size(3))).cuda()
        output = self.decoder(decoder_input, states, self.node_embeddings)
        return output

