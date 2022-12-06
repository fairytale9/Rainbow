#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:28:57 2022

@author: yileichen

Auxiliary replay buffer for RCRL on top of Rainbow
"""

import random
import torch

class replayBuffer():
  def __init__(self, capacity=int(1e6), batch_size=32, discount=1):
    self.capacity = capacity
    self.batch_size = batch_size
    self.discount = discount
    self.segments = {}
    self.cum_reward = torch.zeros(0).to(device=args.device)
    self.traj = []
    self.traj_length = 0
  
  def collect(self, state, action, reward, done):
    self.traj.append((state, action))
    self.traj_length += 1
    self.cum_reward = torch.cat((self.cum_reward, torch.Tensor([0])), 0).to(device=args.device)
    reward_scaling = torch.tensor([self.discount ** (i - 1) for i in range(self.traj_length, 0, -1)], dtype=torch.float32).to(device=args.device)
    self.cum_reward = self.cum_reward + reward * reward_scaling
    if done:
      self.cut_traj(self.traj, self.cum_reward)

  def cut_traj(self, traj, cum_reward):
    cum_reward.tolist()
    for idx, r in enumerate(cum_reward):
      if r in self.segments.keys():
        self.segments[r].append(traj[idx])
      else:
        self.segments[r] = []
    self.empty()
  
  def empty(self):
    self.cum_reward = torch.zeros(0).to(device=args.device)
    self.traj = []
    self.traj_length = 0
    
  def reset(self):
    self.empty()
    self.segments = {}

  def sample(self, batch_size=32):
    return_range = list(self.segments.keys())
    sa_pairs = list(self.segments.values())
    anchor_data = []
    positive_data = []
    negative_data = []
    for i in range(batch_size):
      cum_reward = random.choice(return_range)
      anchor = random.choice(self.segments[cum_reward])
      anchor_data.append(anchor)
      positive_pair = random.choice(self.segments[cum_reward])
      positive_data.append(positive_pair)
      negative_pair = random.choice(sa_pairs)
      negative_data.append(negative_pair)
    return anchor_data, positive_data, negative_data
