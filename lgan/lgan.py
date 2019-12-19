from . import configs
from torch import nn

class L_GAN(nn.Module):
	def __init__(self,dim_z):
		super(L_GAN, self).__init__()
		gen,dis = configs.load_config('lgan')
		self.generator = gen(dim_z)
		self.discriminator = dis(dim_z)

	def gen(self,seed):
		fake = self.generator(seed)
		judge = self.discriminator(fake)
		self.fake = fake
		self.judge = judge
		return judge

	def dis(self,x):
		self.judge = self.discriminator(x)
		return self.judge

def generator_loss(judge):
	return -judge.log().mean()

def discriminator_loss(judge_real,judge_fake):
	return -judge_real.log().mean()-(1-judge_fake).log().mean()
