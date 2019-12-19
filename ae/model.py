from model import base
from torch import nn

class AE(nn.Module):
	def __init__(self,enc,dec):
		super(AE, self).__init__()
		self.encoder = enc
		self.decoder = dec

	def forward(self,x):
		true = x.detach()
		x = self.encoder(x)
		self.z = x.detach()
		x = self.decoder(x)
		# self.rec = x.detach()
		if self.training:
			loss = compute_loss(x, true)
			return x,loss
		else:
			return x

	def encode(self,x):
		return self.encoder(x)

	def decode(self,x):
		return self.decoder(x)


def compute_loss(i,o):

	l = ((i-o)**2).mean(0).sum()
	return l