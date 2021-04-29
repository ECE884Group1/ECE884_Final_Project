#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:50:04 2021

"""


import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pdb
#%%


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
   
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)


    def forward(self, obs):
        # pdb.set_trace()
        q = self.q(obs)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPQnet(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(24,48,48,48),
                 activation=nn.ReLU):
        super().__init__()
        # Observation dimension.
        obs_dim = observation_space.shape[0]
        # Action dimension.
        # pdb.set_trace()
        act_dim = action_space.n
        # Action limit. 
        # act_limit = action_space.high[0]

        # Number of hidden dimension.
        self.num_h = len(hidden_sizes)
        self.hidden = hidden_sizes
        # build main network.
        self.q_net = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        
    def act(self, obs, deterministic=False):
        # pdb.set_trace()
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a
