#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:38:13 2021

"""
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.optim 
import gym
import time
import core as core
import random
import pdb
import torch.optim as optim
from typing import List, Optional
from torch import Tensor
import math


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        # pdb.set_trace()
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
    
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
class DQN:
    def __init__(self,env,q_network=core.MLPQnet,seed=0, 
            replay_size=int(1e6), gamma=0.99, 
            polyak=0.995, lr=1e-3, batch_size=100,hidden_sizes=(24,48,48,48)):
        '''
        Args:
            q_network - DQN network (main and target network).
            env - Environment (GYM).
            replay_size - Replay buffer size.
            gamma - Discount factor.
            polyak - Polyak average parameter. 
            lr - Learning rate. 
            batch_size - Batch size of the current env.
        '''
        # Discount factor.
        self.gamma = gamma
        # Learning rate.
        self.lr = lr
        # Polay (future addition)
        self.polyak = polyak
        # Seed the random value.
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Batch size
        self.batch_size = batch_size
        
        # Observation space.
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n
    
        # Create actor-critic module and target networks
        self.DQ_net = q_network(env.observation_space, env.action_space, hidden_sizes)
        # Traget network.
        self.T_net = deepcopy(self.DQ_net)
    
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.T_net.q_net.parameters():
            p.requires_grad = False
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1, size=replay_size)
        print('R')
        # Count variables (Usefull while experimenting with deeper network).
        self.var_counts = tuple(core.count_vars(module) for module in [self.DQ_net.q_net])
        # Set up optimizers for policy and q-function
        self.q_optimizer = Adam(self.DQ_net.q_net.parameters(), lr=self.lr)
        # Loss function.
        self.mse_loss = torch.nn.MSELoss()
        
    def update(self,data):
        '''
        Args:
            data - batch data.
        '''
        o, a, r, o2, d = data['obs'],\
            data['act'], data['rew'], \
                data['obs2'], data['done']
        # for p in self.T_net.q_net.parameters():
        #     p.requires_grad = False
        # Compute the Q values for the main network.
        # pdb.set_trace()
        a = a.to(torch.int64)
        state_action_values = self.DQ_net.q_net(o).gather(1, a)
        # Create a mask for done task.
        mask = 1 - d
        # Next state Q Values.
        next_state_values = self.T_net.q_net(o2).max(1)[0].detach()
        next_state_values = torch.mul(mask, next_state_values)
        # Expected Q values.
        expected_SA_values = (next_state_values*self.gamma) + r
        
        # Compute the loss.
        # loss = torch.nn.MSELoss(state_action_values, expected_SA_values.unsqueeze(1))
        loss = self.mse_loss(state_action_values, expected_SA_values.unsqueeze(1))
        # pdb.set_trace()
        # Compute the gradient and optimize.
        # for p in self.DQ_net.q_net.parameters():
        #     p.requires_grad = True
        self.DQ_net.q_net.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        
        return loss.detach().item()
    
    
class TDProp:
    def __init__(self,env,q_network=core.MLPQnet,seed=0, 
            replay_size=int(1e6), gamma=0.99, 
            polyak=0.995, lr=1e-3, beta1 = 0.1, beta2 = 0.1, eps = 0.1,
            batch_size=100, hidden_sizes=(24,48,48,48)):
        '''
        Args:
            q_network - DQN network (main and target network).
            env - Environment (GYM).
            replay_size - Replay buffer size.
            gamma - Discount factor.
            polyak - Polyak average parameter. 
            lr - Learning rate. 
            batch_size - Batch size of the current env.
            beta1 - Exponential decay rate of 
        '''
        # Discount factor.
        self.gamma = gamma
        # Learning rate.
        self.lr = lr
        # Polay (future addition)
        self.polyak = polyak
        # Seed the random value.
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Batch size
        self.batch_size = batch_size
        
        # Observation space.
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n
    
        # Create actor-critic module and target networks
        self.DQ_net = q_network(env.observation_space, env.action_space,hidden_sizes)
        # Traget network.
        self.T_net = deepcopy(self.DQ_net)
    
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.T_net.q_net.parameters():
            p.requires_grad = False
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1, size=replay_size)
        print('R')
        # Count variables (Usefull while experimenting with deeper network).
        self.var_counts = tuple(core.count_vars(module) for module in [self.DQ_net.q_net])
        # Set up optimizers for policy and q-function
        # self.q_optimizer = AdamW(self.DQ_net.q_net.parameters(), lr=self.lr)
        self.q_optimizer = TDupdate(self.DQ_net.q_net.parameters(), lr=self.lr)
        # Loss function.
        self.mse_loss = torch.nn.MSELoss()
        
    def update(self,data):
        '''
        Args:
            data - batch data.
        '''
        o, a, r, o2, d = data['obs'],\
            data['act'], data['rew'], \
                data['obs2'], data['done']
        # for p in self.T_net.q_net.parameters():
        #     p.requires_grad = False
        # Compute the Q values for the main network.
        # pdb.set_trace()
        a = a.to(torch.int64)
        state_action_values = self.DQ_net.q_net(o).gather(1, a)
        # Create a mask for done task.
        mask = 1 - d
        # Next state Q Values.
        next_state_values = self.T_net.q_net(o2).max(1)[0].detach()
        next_state_values = torch.mul(mask, next_state_values)
        # Expected Q values.
        expected_SA_values = (next_state_values*self.gamma) + r
        
        # Compute the loss.
        # loss = torch.nn.MSELoss(state_action_values, expected_SA_values.unsqueeze(1))
        loss = self.mse_loss(state_action_values, expected_SA_values.unsqueeze(1))
        # pdb.set_trace()
        # Compute the gradient and optimize.
        # for p in self.DQ_net.q_net.parameters():
        #     p.requires_grad = True
        self.DQ_net.q_net.zero_grad()
        loss.backward()
        # TD error.
        delta = loss.detach().numpy().item()
        # Update the weights. 
        self.q_optimizer.step(delta)
        
        return loss.detach().item()
        
   
class TDupdate(optim.Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(TDupdate, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TDupdate, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, delta, closure=None):
        """Performs a single optimization step.

        Args:
            delta - TD error.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # pdb.set_trace()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                beta1, beta2 = group['betas']
                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            alog_tdprop(params_with_grad,
                    grads,
                    delta,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad,
                    beta1,
                    beta2,
                    group['lr'],
                    group['weight_decay'],
                    group['eps'])

        return loss



def alog_tdprop(params: List[Tensor],
          grads: List[Tensor],
          delta: float,
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        # pdb.set_trace()
        grad = grads[i]
        # Gradient divided by loss.
        grad_delta = grad/delta
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_delta, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class DQN_CE:
    def __init__(self,env,q_network=core.MLPQnet,seed=0, 
            replay_size=int(1e6), gamma=0.99, 
            polyak=0.995, lr=1e-3, beta1 = 0.1, beta2 = 0.1, eps = 0.1,
            batch_size=100,hidden_sizes=(24,48,48,48)):
        '''
        Args:
            q_network - DQN network (main and target network).
            env - Environment (GYM).
            replay_size - Replay buffer size.
            gamma - Discount factor.
            polyak - Polyak average parameter. 
            lr - Learning rate. 
            batch_size - Batch size of the current env.
            beta1 - Exponential decay rate of 
        '''
        # Discount factor.
        self.gamma = gamma
        # Learning rate.
        self.lr = lr
        # Polay (future addition)
        self.polyak = polyak
        # Seed the random value.
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Batch size
        self.batch_size = batch_size
        
        # Observation space.
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n
    
        # Create actor-critic module and target networks
        self.DQ_net = q_network(env.observation_space, env.action_space,hidden_sizes)
        # Traget network.
        self.T_net = deepcopy(self.DQ_net)
    
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.T_net.q_net.parameters():
            p.requires_grad = False
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1, size=replay_size)

        # Count variables (Usefull while experimenting with deeper network).
        self.var_counts = tuple(core.count_vars(module) for module in [self.DQ_net.q_net])
        # Set up optimizers for policy and q-function
        # self.q_optimizer = AdamW(self.DQ_net.q_net.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.DQ_net.q_net.parameters(), lr=self.lr)
        # Loss function.
        self.mse_loss = torch.nn.MSELoss()
        
    def update(self,data):
        '''
        Args:
            data - batch data.
        '''
        o, a, r, o2, d = data['obs'],\
            data['act'], data['rew'], \
                data['obs2'], data['done']
        # for p in self.T_net.q_net.parameters():
        #     p.requires_grad = False
        # Compute the Q values for the main network.
        # pdb.set_trace()
        a = a.to(torch.int64)
        q_val_s = self.DQ_net.q_net(o)
        state_action_values = q_val_s.gather(1, a)
        # Create a mask for done task.
        mask = 1 - d
        # Next state Q Values.
        q_val_next_s = self.T_net.q_net(o2)
        next_state_values = q_val_next_s.max(1)[0].detach()
        next_state_values = torch.mul(mask, next_state_values)
        # Expected Q values.
        expected_SA_values = (next_state_values*self.gamma) + r
        # Compute the loss.
        # pdb.set_trace()
        # loss = torch.nn.MSELoss(state_action_values, expected_SA_values.unsqueeze(1))
        loss = self.mse_loss(state_action_values, expected_SA_values.unsqueeze(1))
        # pdb.set_trace()
        #####################
        # Cross entropy loss.
        #####################
        
        # Probability of choosing action for state s (target network).
        q_val_s_target = self.T_net.q_net(o)
        # state_val_target = q_val_s_target.max(1)[0].detach()
        state_val_target = q_val_s_target.gather(1, a).detach()
        den_ns = (q_val_s_target.detach()).exp()
        sum_den_ns = den_ns.sum(1)
        num_ns = state_val_target.exp()
        num_ns = num_ns.reshape(num_ns.shape[0])
        pi_a_ns = num_ns/sum_den_ns
        
        # Probability of choosing action for state.
        den_s = q_val_s.exp()
        sum_den_s = den_s.sum(1)
        # sum_den_s = sum_den_s.reshape(sum_den_s.shape[0])
        num_s = state_action_values.exp()
        num_s = num_s.reshape(num_s.shape[0])
        pi_a_s = num_s/sum_den_s
        
        check = (den_ns == np.nan).any().item()
        if check == True:
            print('hold')
            pdb.set_trace()
        
        # CE loss.
        # CE_loss = pi_a_ns*pi_a_s.log()
        CE_loss = pi_a_s*pi_a_ns.log()
        CE_loss = -CE_loss
        # KL divergence.
        KL = CE_loss - (-pi_a_s*pi_a_s.log())
        # pdb.set_trace()
        
        # Compute the loss.
        # loss = torch.nn.MSELoss(state_action_values, expected_SA_values.unsqueeze(1))
        # loss = self.mse_loss(state_action_values + 0.0*KL.reshape((KL.shape[0],1)), expected_SA_values.unsqueeze(1))
        loss = self.mse_loss(state_action_values + 0.5*CE_loss.reshape((CE_loss.shape[0],1)), expected_SA_values.unsqueeze(1))
        loss = self.mse_loss(state_action_values, expected_SA_values.unsqueeze(1))
        
        # Total loss
        # total_loss = loss + 0.001*KL.mean()
        # total_loss = loss + 1*CE_loss.mean()
        total_loss = loss
        
        # Compute the gradient and optimize.
        # for p in self.DQ_net.q_net.parameters():
            # p.requires_grad = True
        self.DQ_net.q_net.zero_grad()
        total_loss.backward()
        # for param in self.DQ_net.q_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # Update the weights. 
        self.q_optimizer.step()
        
        return loss.detach().item()