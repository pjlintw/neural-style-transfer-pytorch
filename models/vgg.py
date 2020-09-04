
import torch
import torch.nn as nn

import torchvision.models as models

def VGGConstructor(use_pretrained, device):
	return models.vgg19(pretrained=use_pretrained).features.to(device).eval()

