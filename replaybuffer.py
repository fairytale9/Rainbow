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
  def __init__(self, args, capacity=int(1e6), batch_size=32, discount=1):
    self.device = args.device
    self.capacity = capacity
    self.batch_size = batch_size
    self.discount = discount
    self.segments = []
    self.segment = []
    self.all_data = []
    
  def collect(self, state, action, reward, done):
    self.all_data.append((state, action))
    self.segment.append((state, action))
    if reward > 0:
      self.segments.append(self.segment)
      self.segment.clear()
    if done:
      if self.segment:
        self.segments.append(self.segment)
        self.segment.clear()
    
  def reset(self):
    self.segments.clear()
    self.segment.clear()

  def sample(self, batch_size=32):
    anchor_data = []
    positive_data = []
    negative_data = []
    for i in range(batch_size):
      segment = random.choice(self.segments)
      anchor = random.choice(segment)
      anchor_data.append(anchor)
      positive_pair = random.choice(segment)
      positive_data.append(positive_pair)
      negative_pair = random.choice(self.all_data)
      negative_data.append(negative_pair)
    return anchor_data, positive_data, negative_data
    
