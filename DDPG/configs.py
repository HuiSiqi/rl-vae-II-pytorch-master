from torch import nn
from torch.nn import functional as F
import torch

class Generator(nn.Module):
	def __init__(self, gen,dim_z):
		super(Generator, self).__init__()
		self.m = gen(dim_z)
		self.dim_z = dim_z

	def forward(self, x):
		return self.m(x)

class Discriminator(nn.Module):
	def __init__(self, dis, dim_z):
		super(Discriminator, self).__init__()
		self.m = dis(dim_z)
		self.dim_z=dim_z

	def forward(self, x):
		return self.m(x)

class LGenerator(nn.Module):
	def __init__(self, dim_g,dim_z):
		super(LGenerator, self).__init__()

		self.dim_z = dim_z
		self.dim_g = dim_g
		self.m = nn.Sequential(
			nn.Linear(self.dim_g,128),
			nn.ReLU(),

			nn.Linear(128, 128),
			nn.ReLU(),

			nn.Linear(128,self.dim_z),
		)

	def forward(self, x):
		x = self.m(x)
		return x

class LDiscriminator(nn.Module):
	def __init__(self, dim_z):
		super(LDiscriminator, self).__init__()

		self.dim_z = dim_z

		self.m = nn.Sequential(
			nn.Linear(self.dim_z,128),
			nn.BatchNorm1d(128),
			nn.ReLU(),

			nn.Linear(128,128),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			#512*1*1

			nn.Linear(128,128),
			nn.BatchNorm1d(128),
			nn.ReLU(),

			nn.Linear(128,1),
			nn.BatchNorm1d(1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		x = self.m(x)
		return x

_CONFIG_MAP = {
	'lgan':(LGenerator,LDiscriminator),
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

