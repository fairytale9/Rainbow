# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import DQN


class Agent():
  def __init__(self, args, env, world_model):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.device = args.device
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=self.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    
    self.world_model = world_model
    self.W = nn.Parameter(torch.rand(args.atoms, args.atoms)).to(device=self.device)
    self.cross_entropy_loss = nn.CrossEntropyLoss()

    self.online_net = DQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem, replaybuffer):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    
    # Compute return based contrastive loss
    anchor_data, pos_data = replaybuffer.sample()
    #contrastive_loss = self.compute_contrastive_loss(anchor_data, pos_data, self.online_net)
    anchor_er = self.batch_expected_reward(anchor_data)
    pos_er = self.batch_expected_reward(pos_data)
    diff_er = abs(anchor_er - pos_er)
    true_pos_pairs = [i for i in diff_er if i < 5]
    true_pos_number = len(true_pos_pairs)
    print(true_pos_number)
    
    #rcrl_loss = loss
    
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
    
  # Added functions for RCRL
  def compute_contrastive_loss_v0(self, anchor, pos, neg, embedding_network):
    anchor_sa = self._construct_batched_data(anchor)
    pos_sa = self._construct_batched_data(pos)
    neg_sa = self._construct_batched_data(neg)
    pos_logits = self._compute_logits(anchor_sa, pos_sa, embedding_network)
    neg_logits = self._compute_logits(anchor_sa, neg_sa, embedding_network)
    pos_logits = pos_logits.unsqueeze(1)
    neg_logits = neg_logits.unsqueeze(1)
    logits = torch.cat((pos_logits, neg_logits), 1)
    labels = torch.zeros(self.batch_size).long().to(device=self.device)
    loss = self.cross_entropy_loss(logits, labels)
    return loss

  def compute_contrastive_loss(self, anchor, pos, embedding_network):
    anchor_s, _ = self._construct_batched_data(anchor)
    pos_s, _ = self._construct_batched_data(pos)
    anchor_rep = embedding_network.forward_state_rep(anchor_s)
    pos_s_rep = embedding_network.forward_state_rep(pos_s)
    anchor_pre = embedding_network.forward_predictor(anchor_s)
    pos_s_pre = embedding_network.forward_predictor(pos_s)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity_1 = cos(anchor_rep, pos_s_pre)
    similarity_2 = cos(pos_s_rep, anchor_pre)
    batch_size = similarity_1.shape[0] * 2
    loss = (torch.sum(similarity_1) + torch.sum(similarity_2)) / batch_size
    return loss

  def _compute_logits(self, anchor_sa, sa, embedding_network):
    anchor_sa_rep = embedding_network.forward_representation(anchor_sa) # sa shape [batch_size, atoms]
    sa_rep = embedding_network.forward_representation(sa)
    Wz = torch.matmul(self.W, anchor_sa_rep.T)
    logits = torch.matmul(sa_rep, Wz)
    logits = torch.diagonal(logits)
    return logits
  
  def _construct_batched_data(self, dict_data):
    s = dict_data['s']
    a = dict_data['a']
    batch_s = torch.cat(s, 0).to(device=self.device)
    batch_a = torch.cat(a, 0).to(device=self.device)
    return batch_s, batch_a
    
# Use world model to obtain expected return
  def expectedValue(self, state, system, sample_size=32):
    total_reward = np.zeros(sample_size)
    for i in range(sample_size):
      self.world_model.reset()
      self.world_model.ale.restoreSystemState(system)
      done = False
      while(not done):
        action = self.act(state)  # Choose an action greedily (with noisy weights)
        state, reward, done = self.world_model.step(action)
        total_reward[i] += reward
    return np.mean(total_reward)

  def batch_expected_reward(self, batch_data):
    batch_size = len(batch_data)
    expected_reward = np.zeros(batch_size)
    for i in range(batch_size):
      state, system = batch_data[i]
      expected_reward[i] = self.expectedValue(state, system)
    return expected_reward
    
    
    
    