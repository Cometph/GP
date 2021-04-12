import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
import pickle,time,sys

class LSTHM(nn.Module):
    '''
    LSTHM: An optimized LSTM network which considers the multimodal
    information of the previous time at the current time
    '''

    def __init__(self, dim_input, dim_hidden, dim_Z):
        super(LSTHM, self).__init__()
        # Z: the multimodal fusion information obtained by cross modal attention mechanism
        self.d_i = dim_input
        self.d_h = dim_hidden
        self.d_z = dim_Z

        # output dim: 4*hidden dim->input/output/forget gate and cell status
        # function(W * Xt + U * Ht-1 + V * Zt-1)
        self.W = nn.Linear(self.d_i, 4 * self.d_h)
        self.U = nn.Linear(self.d_h, 4 * self.d_h)
        self.V = nn.Linear(self.d_z, 4 * self.d_h)

    def __call__(self, x, cmt1,hmt1, zmt1):
        # x: time t input;cmt1:time t-1 cell status;
        # hmt1:time t-1 hidden status;zmt1:time t-1 fusion tensor
        # m: modality->language,vision,acoustic; t1: t-1
        return self.forward(x, cmt1,hmt1, zmt1)

    def forward(self, x, cmt1,hmt1, zmt1):
        # function(W * Xt + U * Ht-1 + V * Zt-1)
        input_affine = self.W(x)
        output_affine = self.U(hmt1)
        hybrid_affine = self.V(zmt1)

        # print("input_affine shape:", input_affine.shape)
        # print("output_affine:", output_affine.shape)
        # print("hybrid_affine:", hybrid_affine.shape)

        sums = input_affine + output_affine + hybrid_affine

        # input gate
        i_t = torch.sigmoid(sums[:, :self.d_h])
        # output gate
        o_t = torch.sigmoid(sums[:, self.d_h:2 * self.d_h])
        # forget gate
        f_t = torch.sigmoid(sums[:, 2 * self.d_h:3 * self.d_h])
        # candidate cell status
        ch_t = torch.tanh(sums[:, 3 * self.d_h:])
        # cell status
        c_t = f_t * cmt1 + i_t * ch_t
        # hidden status
        h_t = torch.tanh(c_t) * o_t

        return c_t, h_t
