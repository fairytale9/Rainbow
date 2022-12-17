#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary replay buffer for RCRL on top of Rainbow
"""

import random
import torch
from torch.nn import functional as F
import collections

class replayBuffer():
  def __init__(self, args, action_space, capacity=int(1e3), discount=1):
    self.device = args.device
    self.action_space = action_space
    self.capacity = capacity
    self.batch_size = args.batch_size
    self.discount = discount
    self.segments = []   #or collections.deque(maxlen=self.capacity)
    self.segment = []
    self.all_data = []
    
  def collect(self, state, true_state, reward, done):
    self.all_data.append((state, true_state))
    self.segment.append((state, true_state))
    if reward > 0:
      self.segments.append(self.segment)
      self.segment = []
    if done:
      if self.segment:
        self.segments.append(self.segment)
        self.segment = []

  def collect_1(self, state, action, reward, done):
    self.all_data.append((state, action))
    self.segment.append((state, action))
    if reward > 0:
      self.segments.append(self.segment)
      self.segment = []
    if done:
      if self.segment:
        self.segments.append(self.segment)
        self.segment = []
    
  def reset(self):
    self.segments = []
    self.segment = []

  def sample(self):
    batch_size = self.batch_size
    anchor_data = []
    positive_data = []
    for i in range(batch_size):
      segment = random.choice(self.segments)
      anchor = random.choice(segment)
      anchor_data.append(anchor)
      positive_pair = random.choice(segment)
      positive_data.append(positive_pair)
    return anchor_data, positive_data  

  def sample_v1(self):
    batch_size = self.batch_size
    anchor_data = {'s': [], 'a': []}
    positive_data = {'s': [], 'a': []}
    for i in range(batch_size):
      segment = random.choice(self.segments)
      anchor = random.choice(segment)
      s, a = self._process_before_sample(anchor)
      anchor_data['s'].append(s)
      anchor_data['a'].append(a)
      positive_pair = random.choice(segment)
      s, a = self._process_before_sample(positive_pair)
      positive_data['s'].append(s)
      positive_data['a'].append(a)
    return anchor_data, positive_data

  def sample_v0(self):
    batch_size = self.batch_size
    anchor_data = {'s': [], 'a': []}
    positive_data = {'s': [], 'a': []}
    negative_data = {'s': [], 'a': []}
    for i in range(batch_size):
      segment = random.choice(self.segments)
      anchor = random.choice(segment)
      s, a = self._process_before_sample(anchor)
      anchor_data['s'].append(s)
      anchor_data['a'].append(a)
      positive_pair = random.choice(segment)
      s, a = self._process_before_sample(positive_pair)
      positive_data['s'].append(s)
      positive_data['a'].append(a)
      negative_pair = random.choice(self.all_data)
      s, a = self._process_before_sample(negative_pair)
      negative_data['s'].append(s)
      negative_data['a'].append(a)      
    return anchor_data, positive_data, negative_data
  
  def _process_before_sample(self, sa_pair):
    s, a = sa_pair
    s = torch.unsqueeze(s, 0)
    a = torch.tensor(a)
    a = F.one_hot(a, num_classes=self.action_space)
    a = torch.unsqueeze(a, 0)
    return s, a
    '''
    def sample(self):
      positive_pairs = []
      for i in range(2 * self.batch_size):
        segment = random.choice(self.segments)
        s_1, s_2 = random.sample(segment, 2)
        v_1 = expectedValue()
        v_2 = expectedValue()
        if abs(v_1-v_2)<10:
          positive_pairs.append(s_1, s_2)
      print(len(positive_pairs))
      return positive_pairs
          '''