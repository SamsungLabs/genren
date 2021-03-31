import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, numpy.random as npr
import os, sys, torchvision.utils as TU

class VAE_encoder(nn.Module):

	def __init__(self, init_network, init_out_size, final_out_size):
		super(VAE_encoder, self).__init__()
		self.f = init_network # Don't end with linear layer
		self.g_mu = nn.Linear(init_out_size, final_out_size)
		self.g_logvar = nn.Linear(init_out_size, final_out_size)

	def forward(self, x):
		"""
		Map x to (z(mu,sigma), mu(x), logvar(x))
		"""
		intermed = self.f(x)
		mu = self.g_mu(intermed)
		logvar = self.g_logvar(intermed) 
		stddev = torch.exp(0.5 * logvar) + 1e-4 # Always some injected noise
		noise = torch.randn_like(stddev)
		return (mu + noise*stddev), mu, logvar


#
