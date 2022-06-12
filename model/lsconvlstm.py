#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from model.lsconv import LSConv

# In[2]:


class LSConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(LSConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2
        self.bias = bias
        self.conv = LSConv(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias,)
        
        self.w_c_i = nn.Parameter(torch.Tensor(1, self.hidden_dim, 1, 1))
        nn.init.xavier_uniform_(self.w_c_i)
        self.w_c_f = nn.Parameter(torch.Tensor(1, self.hidden_dim, 1, 1))
        nn.init.xavier_uniform_(self.w_c_f)
        self.w_c_o = nn.Parameter(torch.Tensor(1, self.hidden_dim, 1, 1))
        nn.init.xavier_uniform_(self.w_c_o)

    def forward(self, input_tensor, node_embeddings, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined, node_embeddings)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i+c_cur*self.w_c_i)
        f = torch.sigmoid(cc_f+c_cur*self.w_c_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        o = torch.sigmoid(cc_o+c_next*self.w_c_o)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# In[3]:


class LSConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
       predit next image and last states of the ConvLSTM
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, batch_first=True, bias=False,):
        super(LSConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.bias = bias
        self.cell = LSConvLSTMCell(input_dim=self.input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          bias=self.bias,)
    
    def forward(self, input_tensor, node_embeddings, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        h, c = hidden_state
        
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell(cur_layer_input[:, t, :, :, :], node_embeddings, [h, c])
            output_inner.append(h)
            
        layer_output = torch.stack(output_inner, dim=1)
        return layer_output, [h, c]

    def _init_hidden(self, batch_size, image_size):
        init_states = self.cell.init_hidden(batch_size, image_size)
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

