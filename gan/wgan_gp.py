from . import configs
from torch import nn
import torch
from torch.autograd import grad

class WGAN_GP(nn.Module):
	def __init__(self,dim_g,dim_sample,n_conv=4,gamma=10,config='wgan_gp'):
		super(WGAN_GP, self).__init__()
		gen,dis = configs.load_config(config)
		self.generator = gen(dim_g,output_channels=dim_sample,n_upsamplings=n_conv)
		self.discriminator = dis(input_channels=dim_sample,n_downsamplings=n_conv)
		self.gamma = gamma
		self.register_buffer('dim_g',torch.Tensor([dim_g]).int())
		self.register_buffer('dim_s',torch.Tensor([dim_sample]).int())

	def gen(self,seed):
		self.fake = self.generator(seed)
		return self.fake

	def dis(self,x):
		self.judge = self.discriminator(x)
		return self.judge

	def forward(self, x):
		if self.training:
			seed, true = x
			self.fake_f = self.gen(seed.view(-1,self.dim_g.item(),1,1))
			judge_fake = self.dis(self.fake_f)
			judge_real = self.dis(true)
			gen_loss = generator_loss(judge_fake)
			gp = gradient_penalty(self.discriminator, true,self.fake_f)
			dis_loss = discriminator_loss(judge_real, judge_fake) + gp*self.gamma
			return gen_loss,dis_loss,self.fake_f
		else:
			self.fake_f = self.gen(x)
			return self.fake_f.detach()

def generator_loss(judge_fake):
	return -judge_fake.mean()

def discriminator_loss(judge_real,judge_fake):
	return -judge_real.mean()+judge_fake.mean()

def gradient_penalty(model,real,fake):
	bs = real.size()[0]
	real_dims = len(real.size())
	alpha = torch.rand([bs]+[1]*(real_dims-1))
	alpha = alpha.expand(real.size())
	alpha = alpha.to(real.device)

	interpolates = alpha*real+((1-alpha)*fake)

	judge_inter = model(interpolates)
	gradient = grad(outputs=judge_inter,inputs=interpolates,
	                grad_outputs=torch.ones_like(judge_inter),
	                create_graph=True,retain_graph=True,only_inputs=True)

	gp = (gradient[0].view(bs,-1).norm(2,dim=1)-1)**2

	return gp.mean()
