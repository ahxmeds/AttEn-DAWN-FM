import torch 
import os 
import numpy as np
#%%
def pad_zeros_at_front(num, N):
    return  str(num).zfill(N)

def get_gaussian_noise_std(data, percent_noise):
    # get std for a batch
    # for each image, std is decided based on (s%)*(data.max() - data.min())
    # where s = percent_noise
    data_flat = data.view(data.size(0), -1)
    data_max, _ = torch.max(data_flat, dim=1)
    data_min, _ = torch.min(data_flat, dim=1)
    data_range = data_max - data_min
    std = percent_noise*data_range
    return std

def sample_conditional_pt(x0, x1, t, sigma):
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon, x1-x0

def sample_conditional_pt_sym(x0, x1, t, sigma):
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    gamma = 2*torch.sqrt(t*(1-t))
    mu_t = gamma * x1 + (1-gamma)*x0
    gammap = -(2*t - 1)/(2*(-t*(t - 1))**(1/2)) 
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon, gammap*(x1-x0)


