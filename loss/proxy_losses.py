'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def proxy_synthesis(input_l2, proxy_l2, target, ps_alpha, ps_mu):
    '''
    input_l2: [batch_size, dims] l2-normalized embedding features
    proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    target: [batch_size] Note that adjacent labels should be different (e.g., [0,1,2,3,4,5,...])
    ps_alpha: alpha for beta distribution
    ps_mu: generation ratio (# of synthetics / batch_size)
    '''

    input_list = [input_l2]
    proxy_list = [proxy_l2]
    target_list = [target]

    ps_rate = np.random.beta(ps_alpha, ps_alpha)

    input_aug = ps_rate * input_l2 + (1.0 - ps_rate) * torch.roll(input_l2, 1, dims=0)
    proxy_aug = ps_rate * proxy_l2[target,:] + (1.0 - ps_rate) * torch.roll(proxy_l2[target,:], 1, dims=0)
    input_list.append(input_aug)
    proxy_list.append(proxy_aug)
    
    n_classes = proxy_l2.shape[0]
    pseudo_target = torch.arange(n_classes, n_classes + input_l2.shape[0]).cuda()
    target_list.append(pseudo_target)

    embed_size = int(input_l2.shape[0] * (1.0 + ps_mu))
    proxy_size = int(n_classes + input_l2.shape[0] * ps_mu)
    input_large = torch.cat(input_list, dim=0)[:embed_size,:]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size,:]
    target = torch.cat(target_list, dim=0)[:embed_size]
    
    input_l2 = F.normalize(input_large, p=2, dim=1)
    proxy_l2 = F.normalize(proxy_large, p=2, dim=1)

    return input_l2, proxy_l2, target


class Norm_SoftMax(nn.Module):
    def __init__(self, input_dim, n_classes, scale=23.0, ps_mu=0.0, ps_alpha=0.0):
        super(Norm_SoftMax, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))
        
        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
        

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)
        
        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)

        sim_mat = input_l2.matmul(proxy_l2.t())
        
        logits = self.scale * sim_mat
        
        loss = F.cross_entropy(logits, target)
        
        return loss


class Proxy_NCA(nn.Module):
    def __init__(self, input_dim, n_classes, scale=10.0, ps_mu=0.0, ps_alpha=0.0):
        super(Proxy_NCA, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))
        
        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
    
    
    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)

        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)
 
        dist_mat = torch.cdist(input_l2, proxy_l2) ** 2
        dist_mat *= self.scale
        pos_target = F.one_hot(target, dist_mat.shape[1]).float()
        loss = torch.mean(torch.sum(-pos_target * F.log_softmax(-dist_mat, -1), -1))

        return loss


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, sz_embed, nb_classes, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

