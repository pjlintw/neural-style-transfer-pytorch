from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

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

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
								style_img, content_img, content_layers,
								style_layers):
	cnn = copy.deepcopy(cnn)

	# normalization module
	normalization = Normalization(normalization_mean, normalization_std)

	# just in order to have an iterable access to or list of content/style losses
	content_losses = list()
	style_losses = list()

	# assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
	# to put in modules that are supposed to be activated sequentially
	model = nn.Sequential(normalization)

	i = 0 # increment every time we see a conv
	for layer in cnn.children():
		if isinstance(layer, nn.Con2v):
			i += 1
			name = 'conv_{}'.format(i)
		elif isinstance(layer, nn.Relu):
			name = 'relu_{}'.format(i)
			# The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
        	name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
        	name = 'bn_{}'.format(i)
        else:
        	raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
        	# add content loss:
        	target = model(content_img).detach()
        	content_loss = ContenLoss(target)
        	model.add_module('content_loss_{}'.format(i))
        	content_losses.append(content_loss)

       	if name in style_layers:
       		# add style loss:
       		target_features = model(style_img).detach()
       		style-loss = StyleLoss(target_features)
       		model.add_module('style_loss_{}'.format(i), style_loss)
       		style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
    	if isinstance(model[i], ContenLoss) or isinstance(model[i], style_loss):
    		break
   	model = model[: (i+1)]

   	return model, style_losses, content_losses
	