
import torch

def get_vgg_norm_mean_std():
	cnn_norm_mean = torch.tensor([0.485, 0.456, 0.406])
	cnn_norm_std = torch.tensor([0.229, 0.224, 0.225])
	return cnn_norm_mean, cnn_norm_std

class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		# .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, img):
		# normalize img
		return (img - self.mean) / self.std