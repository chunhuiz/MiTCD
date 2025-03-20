import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out



def Gaussian_kernel(sigma,t,T):
    '''
    input: sigma (kernel width), t (time index), T (overall length)
    output: value list of shape 1*T
    '''
    value = torch.zeros([1,T])
    for i in range(T):
        value[0,i]=(0.39894228/sigma)*torch.exp(-(i-t)*(i-t)/(2*sigma*sigma))
    return value

def Gaussian_kernel_fix(sigma,T):
    '''
    input: sigma (kernel width),  T (overall length)
    output: value list of shape 2*T-1
    release the need to conpute every time!
    '''
    value = torch.zeros([1,2*T-1])
    for i in range(2*T-1):
        value[0,i]=(0.39894228/sigma)*torch.exp(-(i-T)*(i-T)/(2*sigma*sigma))
    return value



def GKE(X,sigma,T):
    '''
    input: Discrete Time series X ( 0 0 1 1 0 1) of shope 1*T , sigma (kernel width), t (time index), T (overall length)
    output: Gaussian Embedding shape 1*T
    '''
    embed_value = torch.zeros([1,T])
    vectors=Gaussian_kernel_fix(sigma,T)
    for i in range(T):
        if X[0,i]>0.5:
            embed_value=embed_value+vectors[:,T-i-1:2*T-i-1]
    return embed_value





class CAGKE_1(nn.Module):
    '''
    Contexual Adaptative Gaussain Kernel Embedding
    input: Discrete Time series X ( 0 0 1 1 0 1) of shope 1*T

    processing: Gausssian Embedding vector of shaoe D*T

    output: Weighted sum of Gaussian Embedding vector of shaoe 1*T
    Gaussian Noise added


    '''
    def __init__(self, in_length,embed_dim,sigma_min,sigma_max,noise_sigma):
        super(CAGKE_1, self).__init__()
        self.in_length = in_length
        self.embed_dim = embed_dim
        sigma = torch.tensor(np.linspace(sigma_min,sigma_max,self.embed_dim))
        scale_weight = torch.ones([1,self.embed_dim])

        self.learnable_sigma = nn.Parameter(sigma)
        self.learnable_weight = nn.Parameter(scale_weight)
        self.noise = noise_sigma*torch.randn(1,in_length)


    def forward(self, X):
        embed_vector=torch.zeros([self.embed_dim,self.in_length]).to(X.device)
        # learnable_sigma = self.learnable_sigma.to(X.device)
        # learnable_weight = self.learnable_weight.to(X.device)

        for i in range(self.embed_dim):

            embed_vector[[i],:]=GKE(X,self.learnable_sigma[i],self.in_length).to(X.device)

        return torch.mm(F.softmax(self.learnable_weight,dim=1),embed_vector)+self.noise


class CAGKE_learnable(nn.Module):
    '''
    Contexual Adaptative Gaussain Kernel Embedding

    input: Discrete Time series X ( 0 0 1 1 0 1) of shope 1*T

    processing: Gausssian Embedding vector of shaoe D*T

    output: Weighted sum of Gaussian Embedding vector of shaoe 1*T
    Gaussian Noise added


    '''
    def __init__(self, in_length,embed_dim,sigma_min=0.4,sigma_max=4,noise_sigma=0.01):
        super(CAGKE_learnable, self).__init__()
        self.in_length = in_length
        self.embed_dim = embed_dim
        sigma = torch.tensor(np.linspace(sigma_min,sigma_max,self.embed_dim))
        scale_weight = torch.ones([1,self.embed_dim])

        self.learnable_sigma = nn.Parameter(sigma)
        self.learnable_weight = nn.Parameter(scale_weight)
        self.noise_sigma = noise_sigma

    def forward(self, X, norm = True):
        embed_vector=torch.zeros([self.embed_dim,self.in_length]).to(X.device)
        # learnable_sigma = self.learnable_sigma.to(X.device)
        # learnable_weight = self.learnable_weight.to(X.device)

        for i in range(self.embed_dim):

            # ensure that sigma>0
            embed_vector[[i],:]=GKE(X,torch.abs(self.learnable_sigma[i]),self.in_length).to(X.device)

        psedu_conti = torch.mm(F.softmax(self.learnable_weight,dim=1),embed_vector) + self.noise_sigma*torch.randn(1,self.in_length).to(X.device)


        if norm:
            output=(psedu_conti-torch.min(psedu_conti))/(torch.max(psedu_conti)-torch.min(psedu_conti))

            return output
        else:

            return psedu_conti


def sig_linspace(start,end,steps):

    '''
    generate linspace series
    '''
    output = torch.ones(steps).to(start.device)
    # output[0] = start.data
    # output[-1] = end.data
    length_steps = (end-start)/(steps-1)
    for i in range(steps):
        output[i] = start + i*length_steps

    return output




class CAGKE_learnable_minmax(nn.Module):
    '''
    Contexual Adaptative Gaussain Kernel Embedding

    input: Discrete Time series X ( 0 0 1 1 0 1) of shope 1*T

    processing: Gausssian Embedding vector of shaoe D*T

    output: Weighted sum of Gaussian Embedding vector of shaoe 1*T
    Gaussian Noise added


    '''
    def __init__(self, in_length,embed_dim,sigma_min=0.4,sigma_max=4,noise_sigma=0.01):
        super(CAGKE_learnable_minmax, self).__init__()
        self.in_length = in_length
        self.embed_dim = embed_dim
        sigma_min = torch.tensor(sigma_min)
        sigma_max = torch.tensor(sigma_max)

        scale_weight = torch.ones([1,self.embed_dim])

        self.learnable_sigma_min = nn.Parameter(sigma_min)
        self.learnable_sigma_max = nn.Parameter(sigma_max)

        self.learnable_weight = nn.Parameter(scale_weight)
        self.noise_sigma = noise_sigma

    def forward(self, X, norm = True):
        embed_vector=torch.zeros([self.embed_dim,self.in_length]).to(X.device)
        sigma_list = sig_linspace(start = self.learnable_sigma_min, end = self.learnable_sigma_max, steps= self.embed_dim)

        for i in range(self.embed_dim):

            # ensure that sigma>0
            embed_vector[[i],:]=GKE(X,torch.abs(sigma_list[i]),self.in_length).to(X.device)

        psedu_conti = torch.mm(F.softmax(self.learnable_weight,dim=1),embed_vector) + self.noise_sigma*torch.randn(1,self.in_length).to(X.device)


        if norm:
            output=(psedu_conti-torch.min(psedu_conti))/(torch.max(psedu_conti)-torch.min(psedu_conti))

            return output
        else:

            return psedu_conti



class CAGKE_fix(nn.Module):
    '''
    Contexual Adaptative Gaussain Kernel Embedding

    input: Discrete Time series X ( 0 0 1 1 0 1) of shope 1*T

    processing: Gausssian Embedding vector of shaoe D*T

    output: Weighted sum of Gaussian Embedding vector of shaoe 1*T

    '''
    def __init__(self, in_length,embed_dim,sigma_min=0.4,sigma_max=3,noise_sigma=0.01):
        super(CAGKE_fix, self).__init__()
        self.in_length = in_length
        self.embed_dim = embed_dim
        self.sigma = torch.tensor(np.linspace(sigma_min,sigma_max,self.embed_dim))
        self.weight = torch.ones([1,self.embed_dim])

        self.noise = noise_sigma * torch.randn(1, in_length)

    def forward(self, X,norm = True):
        embed_vector=torch.zeros([self.embed_dim,self.in_length]).to(X.device)
        sigma = self.sigma.to(X.device)
        weight = self.weight.to(X.device)

        for i in range(self.embed_dim):

            embed_vector[[i],:]=GKE(X,sigma[i],self.in_length).to(X.device)

        psedu_conti = torch.mm(F.softmax(weight,dim=1),embed_vector)+self.noise

        if norm:
            output = (psedu_conti - torch.min(psedu_conti)) / (torch.max(psedu_conti) - torch.min(psedu_conti))

            return output
        else:

            return psedu_conti




