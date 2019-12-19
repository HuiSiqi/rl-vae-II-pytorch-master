from torch import nn
from torch.nn import functional as F
import torch, utils


class Generator(nn.Module):
	def __init__(self, gen, dim_g, dim_z):
		super(Generator, self).__init__()
		self.m = gen(dim_g, dim_z)
		self.dim_z = dim_z

	def forward(self, x):
		return self.m(x)


class Discriminator(nn.Module):
	def __init__(self, dis, dim_z):
		super(Discriminator, self).__init__()
		self.m = dis(dim_z)
		self.dim_z = dim_z

	def forward(self, x):
		return self.m(x)

class LWGC(nn.Module):
	def __init__(self, dim_g, dim_z):
		super(LWGC, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(dim_g, 256),
			nn.ReLU(),
			nn.Linear(256, 4 * 4 * 256),
			nn.BatchNorm1d(4 * 4 * 256),
			nn.ReLU(),
		)
		self.m = nn.Sequential(
			# 256*4*4
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			# 1024*4*4
			# todo 256*8*8
			nn.Upsample((8, 8), mode='bilinear', align_corners=True),
			nn.Conv2d(256, 256, 5, 1, 2),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			# todo 256*8*8
			nn.Upsample((16, 16), mode='bilinear', align_corners=True),
			nn.Conv2d(256, 256, 5, 1, 2),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			# todo 256*16*16
			nn.Upsample((32, 32), mode='bilinear', align_corners=True),
			nn.Conv2d(256, 128, 5, 1, 2),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 64, 5, 1, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, dim_z, 3, 1, 1),
			nn.BatchNorm2d(dim_z),
		)
		self.apply(utils.weights_init)

	def forward(self, x):
		x = self.fc1(x)
		x = self.m(x.view(-1, 256, 4, 4))
		return x


class LWDC(nn.Module):
	def __init__(self, dim_z):
		super(LWDC, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(256, 1024),
			nn.ReLU(),
			nn.Linear(1024, 1),
		)
		self.m = nn.Sequential(
			# 32*32
			nn.Conv2d(dim_z, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			# 16*16
			nn.Conv2d(64, 128, 3, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			# 8*8
			nn.Conv2d(128, 256, 3, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			# 4*4
			nn.Conv2d(256, 256, 4, 1, 0),
			nn.BatchNorm2d(256),
			nn.ReLU(),
		)
		self.apply(utils.weights_init)

	def forward(self, z):
		z = self.m(z)
		z = self.fc(z.view(-1, 256))
		return z


from torchvision import models


class Dis256(nn.Module):
	def __init__(self, dim_z):
		super(Dis256, self).__init__()
		self.fc = nn.Linear(1024, 1)
		self.m = nn.Sequential(
			# todo 3*256*256
			nn.Conv2d(3, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			# todo 32*256*256
			nn.Conv2d(32, 64, 3, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			# todo 64*128*128
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64 * 2, 3, 2, 1),
			nn.BatchNorm2d(64 * 2),
			nn.ReLU(),
			# todo 128*64*64
			nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1),
			nn.BatchNorm2d(64 * 2),
			nn.ReLU(),
			nn.Conv2d(64 * 2, 64 * 4, 3, 2, 1),
			nn.BatchNorm2d(64 * 4),
			nn.ReLU(),
			# todo 256*32*32
			nn.Conv2d(64 * 4, 64 * 4, 3, 1, 1),
			nn.BatchNorm2d(64 * 4),
			nn.ReLU(),
			nn.Conv2d(64 * 4, 64 * 8, 3, 2, 1),
			nn.BatchNorm2d(64 * 4),
			nn.ReLU(),
			# todo 256*32*32
			nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1),
			nn.BatchNorm2d(64 * 2),
			nn.ReLU(),
			nn.Conv2d(64 * 2, 64 * 4, 3, 2, 1),
			nn.BatchNorm2d(64 * 4),
			nn.ReLU(),
		)
		self.apply(utils.weights_init)

	def forward(self, z):
		z = self.m(z)
		z = self.fc(z.view(-1, 256))
		return z


_CONFIG_MAP = {
	'wgan': (WGC, WDC),
}


def load_config(name):
	"""Load a particular configuration
	Returns:
	(encoder, transition, decoder) A tuple containing class constructors
	"""
	if name not in _CONFIG_MAP.keys():
		raise ValueError("Unknown config: %s", name)
	return _CONFIG_MAP[name]


__all__ = ['load_config']

