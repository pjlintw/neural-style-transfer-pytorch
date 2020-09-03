from __future__ import print_function

import torch

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

device = torch.device('cude', if torch.cude.is_avaliable() else 'cpu')

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
	image = Image.open(image_name)
	image = loader(image).unsqueeze(0)
	return image.to(device, torch.float)

def save_image_from_tensor(tensor, output_path):
	pass
	