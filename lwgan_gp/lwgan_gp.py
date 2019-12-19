from . import configs
from torch import nn

class LWGAN(nn.Module):
	def __init__(self,dim_g,dim_z):
		super(LWGAN, self).__init__()
		gen,dis = configs.load_config('lwgan')
		self.generator = gen(dim_g,dim_z)
		self.discriminator = dis(dim_z)

	def gen(self,seed):
		self.fake = self.generator(seed)
		return self.fake

	def dis(self,x):
		self.judge = self.discriminator(x)
		return self.judge

def generator_loss(judge):
	return -judge.mean()

def discriminator_loss(judge_real,judge_fake):
	return -judge_real.mean()+judge_fake.mean()

def clip_weight(m,c):
	for p in m.parameters():
		p.data.clamp_(-c,c)
