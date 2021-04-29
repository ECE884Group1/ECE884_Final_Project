#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:16:11 2021

"""
# Library imports.

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import os

import torch
import RL_algo as RL
import pdb
import pickle

    
#%% Testing the network.

def test_DQN(agent,env):
    """
    Testing the DQN network.
    Args:
        agent - The trained agent.
        env - the environment.
    """
    # Save the states (per episode).
    state_test = {}
    # Save the actions (outer loop).
    act_test_save = {}
    # Test episode.
    test_episodes = 100
    # Reward save.
    r_ep = []
    # Steps done.
    steps_done = 0
    # Time step
    t_eps = 500
    
    for ep in range(1,test_episodes):
        # Save the actions (inner loop).
        act_save_il = []
        # Save the states (inner loop).
        state_train_il = []
        
        # Initialize the environment and state.
        state = env.reset()
        # Save the start state.
        state_train_il.append(state)
        # Inner loop.
        # Accumulate reward.
        rew_acc = []
        
        for t in range(t_eps):
            # Select the action. 
            action = select_action_CEL(state,steps_done,test=True)
            action = action.item()
            # Update the global counter.
            steps_done +=1
            # Save the action.
            act_save_il.append(action)
            # Next state based on the action taken. 
            next_state, reward, done, _ = env.step(action)
            # Accumulate reward.
            rew_acc.append(reward)
            # pdb.set_trace()
            # Store experience to replay buffer
            agent.replay_buffer.store(state, action, reward, next_state, done)
            # Save the next state.
            state_train_il.append(next_state)
            # Update the state.
            state = np.copy(next_state)
                    
            if done:
                break
            
        # if ep % 10 == 0:
        #     print('Testing, episode {}, reward {}'.format(ep,np.sum(rew_acc)))

            
        # Save the reward for an episode.
        r_ep.append(np.sum(rew_acc))
        # Save the action.
        act_test_save[ep] = act_save_il
        # Save the state.
        state_test[ep] = np.vstack(state_train_il)
    
    print('Testing, episode {}, reward {}'.format(ep,np.mean(r_ep)))
    return r_ep, act_test_save, state_test
        

#%% Cross-entropy loss.
#######################


def select_action_CEL(state,steps_done,test=False):
    r"""Selects action for the corresponding environment.

    Args:
        state (numpy array): state vector for the environment.
        cnt (int): Global counter.
    """
    # State to tensor and reshape.
    state = torch.as_tensor(state, dtype=torch.float32)
    state = state.reshape((1,state.shape[0]))
    
    # Random sample.
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    if not test:
        if sample > eps_threshold:
            # pdb.set_trace()
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return agent_CE.DQ_net.q_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(num_action)]], dtype=torch.long)
    else:
        return agent_CE.DQ_net.q_net(state).max(1)[1].view(1, 1)
    
#%% Agent and parameter
# Import gym environment. 
env_name = 'CartPole-v1'
env = gym.make(env_name).unwrapped
# Actions.
num_action = env.action_space.n
# Batch size.
batch_size = 32
# Discount factor.
gamma = 0.99
# Start, end and decay rate of epsilon greedy sampling.
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 200
# Update target per epsiode.
TARGET_UPDATE = 50
# Update every episode.
update_every = 1
# DQN_CE agent.
agent_CE = RL.DQN_CE(env, batch_size = batch_size, gamma=gamma,\
               lr= 1e-3)

# Number of episodes.
episodes = 1000
# Number of time steps in an episode.
t_eps = 200
# Polay averaging.
polyak = 0.995
#%% Saving 
# Number of states
num_state = len(env.reset())
# Save the states (per episode).
state_train = {}
# Save the actions (outer loop).
act_save = {}
# Loss 
loss_str = []
# Reward global
r_g_DQN_CE = []
# Saving duration.
save_eps = 0
#%% Training loop

# Global counter.
steps_done = 0

# Reward for an episode.
r_ep = 0

for ep in range(0,episodes):
    # Save the actions (inner loop).
    act_save_il = []
    # Save the states (inner loop).
    state_train_il = []
    
    # Initialize the environment and state.
    state = env.reset()
    # Save the start state.
    state_train_il.append(state)
    # Inner loop.
    # Accumulate reward.
    rew_acc = []
    
    for t in count():
        # Select the action. 
        if ep < save_eps:
            action = np.random.randint(0,num_action)
        else:
            action = select_action_CEL(state,steps_done)
            action = action.item()
            # Update the global counter.
            steps_done +=1
        # Save the action.
        act_save_il.append(action)
        # Next state based on the action taken. 
        next_state, reward, done, _ = env.step(action)
        # Accumulate reward.
        rew_acc.append(reward)
        # pdb.set_trace()
        # Store experience to replay buffer
        agent_CE.replay_buffer.store(state, action, reward, next_state, done)
        # Save the next state.
        state_train_il.append(next_state)
        # Update the state.
        state = np.copy(next_state)
        # Update loop.
        if steps_done >= batch_size and ep > save_eps and ep % update_every == 0:
            # pdb.set_trace()
            for j in range(update_every):
                batch = agent_CE.replay_buffer.sample_batch(batch_size)
                batch['obs'] = batch['obs']
                batch['obs2'] = batch['obs2']
                batch['act'] = batch['act']
                batch['rew'] = batch['rew']
                batch['done'] = batch['done']
                # Update the main network.        
                loss_str.append(agent_CE.update(data=batch)) 
                
                # Soft update.
                with torch.no_grad():
                    for p, p_targ in zip(agent_CE.DQ_net.parameters(), agent_CE.T_net.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)
                
        if done or t > 1000:
            break
        
        
        # if steps_done % 100==0:
            # print(np.sum(rew_acc))
    # Save the weights.
    if ep % 5 == 0:
        r_test,_,_ =  test_DQN(agent_CE,env)
        if np.mean(r_test) > r_ep:
            r_ep = np.mean(r_test)
            # Save the weights.
            path = os.getcwd()
            dir_save = path + '/' +str(env_name) + '_DQN_CE'
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)
    
            model_DQN = dir_save + '/Model_DQN_CE'
            # Saving the weights.
            torch.save(agent_CE.DQ_net.q_net.state_dict(), model_DQN)
    
        print('Training, episode {}, reward {}'.format(ep,np.sum(rew_acc)))
    
    r_g_DQN_CE.append(np.sum(rew_acc))
    # Update the target network.
    # Hard update.
    # if ep % TARGET_UPDATE == 0:
        # agent.T_net.q_net.load_state_dict(agent.DQ_net.q_net.state_dict())
        # print('Network Updated')
#%% Data from testing.
path = os.getcwd()
dir_save = path + '/' +str(env_name) + '_DQN_CE_0.5'  + \
            str(agent_CE.DQ_net.num_h) + str(agent_CE.DQ_net.hidden)
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
r_test, act_test, state_test = test_DQN(agent_CE,env)
# Save the reward, state, action.
with open(dir_save + '/reward_test.pickle', 'wb') as handle:
    pickle.dump(r_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/reward_train.pickle', 'wb') as handle:
    pickle.dump(r_g_DQN_CE, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/action_test.pickle', 'wb') as handle:
    pickle.dump(act_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(dir_save + '/state_test.pickle', 'wb') as handle:
    pickle.dump(state_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
model_DQN = dir_save + '/Model_DQN_CE'
# Saving the weights.
# torch.save(agent_CE.DQ_net.q_net.state_dict(), model_DQN)
#%%
'''
import time
idx = np.random.randint(0,len(state_test))
state_sample = state_test[idx]
# Set the initial state.
state = env.reset()
# env.state = state_sample[0,:]
# Get the actions.
# act_sample = act_test[idx]
# frames = []
for x in range(len(state_sample)-1):
    # frames.append(env.render(mode="rgb_array"))
    env.render()
    act = select_action_CEL(state,x,True)
    # Take the action.
    # act = act_sample[x]
    env.step(act)
    time.sleep(.03)
env.close()
# save_frames_as_gif(frames,load_dir,filename = env_name + '.gif')
'''