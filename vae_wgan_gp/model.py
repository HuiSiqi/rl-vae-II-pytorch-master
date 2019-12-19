import torch
from . import configs
from torch import nn,distributions
from .configs import load_config
from .losses import binary_crossentropy
from torch.autograd import grad

class VAE(torch.nn.Module):
	def __init__(self, dim_z, config='vae'):
		super(VAE, self).__init__()
		self.dim_z = dim_z
		enc, dec = load_config(config)
		self.encoder = configs.Encoder(enc,dim_z)
		self.decoder = configs.Decoder(dec,dim_z)
		self.prior = distributions.Normal(0,1)

	def reparam(self, mean, logvar):
		std = logvar.mul(0.5).exp_()
		self.z_mean = mean
		self.z_sigma = std
		eps = self.prior.sample(mean.size()).to(std.device)
		return eps.mul(std).add_(mean)

	def forward(self, x):
		self.z_mean, logvar = self.encoder(x)
		self.z_logsigma = logvar.mul(0.5)

		self.z = self.reparam(self.z_mean, logvar)

		x_dec = self.decoder(self.z)

		return x_dec

	def encode(self, x):
		#with torch.no_grad():
		return self.encoder(x)[0]

	def decode(self,z):
		return self.decoder(z)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.m = nn.Sequential(
			# todo 3*64*64
			nn.Conv2d(3, 32, 5, 2, 2),
			nn.ReLU(),
			#32
			nn.Conv2d(32, 128, 5, 2, 2),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			#16
			nn.Conv2d(128, 64 * 4, 5, 2, 2),
			nn.BatchNorm2d(64 * 4),
			nn.ReLU(),
			#8
			nn.Conv2d(64 * 4, 64 * 4, 5, 2, 2),
			nn.BatchNorm2d(64 * 4),
			nn.ReLU(),
			#4
		)

		self.out_reshape = nn.Sequential(
			nn.Linear(64*4*4*4,512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512,1),
		)

	def forward(self, x):
		x = self.m(x)
		x = self.out_reshape(x.view(-1, 64 * 4*4*4))
		return x

def vae_loss(x_pred, x_true, z_mean, z_logsigma, mse=False):
	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	bs = x_pred.shape[0]
	x_pred = x_pred.view(bs,-1)
	x_true = x_true.view(bs,-1)
	z_mean =  z_mean.view(bs,-1)
	z_logsigma = z_logsigma.view(bs,-1)

	if mse:
		x_reconst_loss = (x_pred - x_true).pow(2).sum(dim=1)
	else:
		x_reconst_loss = -binary_crossentropy(x_true, x_pred).sum(dim=1)
	logvar = z_logsigma.mul(2)
	KLD_element = z_mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.sum(KLD_element, dim=1).mul(-0.5)
	return x_reconst_loss.mean(), KLD.mean()

def generator_loss(judge):
	return -judge.mean()

def discriminator_loss(judge_real,judge_fake):
	return -judge_real.mean()+judge_fake.mean()

def gradient_penalty(model,real,fake):
	fake.requires_grad_(True)
	real.requires_grad_(True)
	real = real.to(fake.device)
	bs = real.size()[0]
	dim = len(real.size())
	alpha = torch.rand(*([bs]+[1]*(dim-1)))
	alpha = alpha.expand(real.size())
	alpha = alpha.to(fake.device)

	interpolates = alpha*real+((1-alpha)*fake)

	judge_inter = model(interpolates)
	gradient = grad(outputs=judge_inter,inputs=interpolates,
	                grad_outputs=torch.ones_like(judge_inter),
	                create_graph=True,retain_graph=True,only_inputs=True)

	gp = (gradient[0].view(bs,-1).norm(2,dim=1)-1)**2

	return gp.mean()

