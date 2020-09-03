from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContenLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input

def gram_matrix(input):
	a, b, c, d = input.size()
	features = input.view(a*b, c*d)

	# compute the gram product
	G = torch.mm(features, features.t())

	# we `normalize` the values of the gram matrix
	# by dividingof of the number of element in each feature maps
	return G.div(a*b*c*d)

class StyleLoss(nn.Module):
	def __init__(self, target_features):
		super(StyleLoss, self).__init__():
		self.target = gram_matrix(target_features).detach()

	def forward(self, input):
		G = gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input