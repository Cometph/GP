import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
import pickle,time,sys

class CrossShareUnit(nn.Module):
    def __init__(self, l_hidden_dim, m_hidden_dim, K, max_len):
        super(CrossShareUnit, self).__init__()
        self.dim_l = l_hidden_dim
        self.dim_m = m_hidden_dim
        self.K = K
        self.max_len = max_len

        self.dropout = nn.Dropout(0.3)

        self.l_fc1 = nn.Linear(l_hidden_dim, 256)
        self.l_fc2 = nn.Linear(256, l_hidden_dim)

        self.m_fc1 = nn.Linear(m_hidden_dim, 128)
        self.m_fc2 = nn.Linear(128, m_hidden_dim)

    def __call__(self, l_hidden, m_hidden):
        # l_hidden: hidden status of language modality
        # m_hidden: hidden status of vision modality or acoustic modality
        return self.forward(l_hidden, m_hidden)

    def forward(self, l_hidden, m_hidden):
        G_l_m = torch.randn(self.dim_l, self.K, self.dim_m,requires_grad=True).cuda()
        G_m_l = torch.randn(self.dim_m, self.K, self.dim_l,requires_grad=True).cuda()

        G_vec_l = self.l_fc2(self.dropout(self.l_fc1(l_hidden)))
        G_vec_m = self.m_fc2(self.dropout(self.m_fc1(m_hidden)))

        # Get share representation
        G_l_m = torch.reshape(G_l_m,shape=[self.dim_l,-1])
        G_l_m = G_l_m.unsqueeze(0)
        G_l_m = G_l_m.repeat(l_hidden.shape[0], 1, 1)

        shared_hidden_l_m = torch.matmul(l_hidden,G_l_m)
        shared_hidden_l_m = torch.reshape(shared_hidden_l_m,shape=[-1, self.max_len * self.K, self.dim_m])

        m_hidden_transpose = m_hidden.permute([0, 2, 1])

        shared_hidden_l_m = torch.tanh(torch.matmul(shared_hidden_l_m, m_hidden_transpose))
        shared_hidden_l_m = torch.reshape(shared_hidden_l_m, shape=[-1, self.max_len, self.K, self.max_len])

        shared_hidden_l_m = shared_hidden_l_m.permute([0, 1, 3, 2])
        shared_hidden_l_m = torch.reshape(shared_hidden_l_m, shape=[-1, self.max_len * self.max_len, self.K])

        G_vec_l = G_vec_l.repeat(1, self.K, 1)

        shared_hidden_l_m = torch.matmul(shared_hidden_l_m, G_vec_l)

        l_vector = torch.reshape(shared_hidden_l_m, shape=[-1, self.max_len * self.max_len, self.dim_l])
        #############################
        G_m_l = torch.reshape(G_m_l,shape=[self.dim_m,-1])
        G_m_l = G_m_l.unsqueeze(0)
        G_m_l = G_m_l.repeat(m_hidden.shape[0], 1, 1)

        shared_hidden_m_l = torch.matmul(m_hidden, G_m_l)
        shared_hidden_m_l = torch.reshape(shared_hidden_m_l, shape=[-1, self.max_len * self.K, self.dim_l])

        l_hidden_transpose = l_hidden.permute([0, 2, 1])

        shared_hidden_m_l = torch.tanh(torch.matmul(shared_hidden_m_l, l_hidden_transpose))
        shared_hidden_m_l = torch.reshape(shared_hidden_m_l, shape=[-1, self.max_len, self.K, self.max_len])

        shared_hidden_m_l = shared_hidden_m_l.permute([0, 1, 3, 2])
        shared_hidden_m_l = torch.reshape(shared_hidden_m_l, shape=[-1, self.max_len * self.max_len, self.K])

        G_vec_m = G_vec_m.repeat(1, self.K, 1)

        shared_hidden_m_l = torch.matmul(shared_hidden_m_l, G_vec_m)

        m_vector = torch.reshape(shared_hidden_m_l, shape=[-1, self.max_len * self.max_len, self.dim_m])
        ############################################
        # Get attention vector
        l_attention_vector = torch.softmax(l_vector, dim=-1)
        m_attention_vector = torch.softmax(m_vector, dim=-1)

        l_hidden_vec = torch.mul(l_attention_vector, l_hidden)###xiu gai???
        m_hidden_vec = torch.mul(m_attention_vector, m_hidden)

        l_hidden = l_hidden + l_hidden_vec
        m_hidden = m_hidden + m_hidden_vec

        return l_hidden, m_hidden, l_attention_vector, m_attention_vector