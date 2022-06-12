#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from torch import nn
import torch.nn.functional as F


# In[2]:


class NodeBasedAttentionModule(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=1, padding=0, norm_num_groups=1, norm_num_channel=1, affine=True):
        
        super(NodeBasedAttentionModule, self).__init__()
        self.spatail_atten = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(norm_num_groups, inchannels, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(norm_num_groups, inchannels, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        atten_value = self.spatail_atten(x)
        atten_output = atten_value * x
        return atten_output


# In[3]:


class LSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, stride=1, padding=0, cheb_k=3):
        super(LSConv, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size_h = kernel_size[0]
        self.kernel_size_w = kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.cheb_k = cheb_k
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size_h, self.kernel_size_w))
        nn.init.xavier_normal_(self.weight)
        
        self.spatial_atten_block = NodeBasedAttentionModule(inchannels=in_channels, outchannels=in_channels)
        if self.bias == True:
            self.b = nn.Parameter(torch.Tensor(self.out_channels,))
            nn.init.xavier_normal_(self.b)
        
#     @torchsnooper.snoop()
    def forward(self, x, node_embeddings):
        b, c, h, w = x.shape
        x = self.spatial_atten_block(x)
        x = x.permute(0, 2, 3, 1).reshape(-1, h*w, c)
        graph_constraint = []
        conv_result = []
        
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        supports = torch.eye(node_num).to(supports.device) + supports
        graph_constraint = torch.einsum("nm,bmc->bnc", supports, x)
        graph_constraint = graph_constraint.permute(0, 2, 1).reshape(-1, c, h, w,)
    
        return F.relu(F.conv2d(graph_constraint, self.weight, stride=self.stride, padding=self.padding,))
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size must be tuple or list of tuples')
        if kernel_size[0] != kernel_size[1]:
            raise ValueError('`kernel_size must have the same size!')
        if kernel_size[0] % 2 == 0:
            raise ValueError('`kernel_size must be odd')

