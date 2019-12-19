#todo ------------------------------------------------------basic packadges-------------------------------------------
import torch,os,utils,numpy as np
from datasets import datasets
from torch.utils.data import DataLoader
from torch import optim
from . import utils as env_utils
#todo -----------------------load model packages--------------------------
from vae_wgan import vae_wgan
from vae_wgan_gp import vae_wgan_gp
from vae_gan import vae_gan
from ae import ae
from lgan import lgan

class env():
	def __init__(self,args):
		#todo init reward weight
		self.wmse = args.wmse
		self.wpsnr = args.wpsnr
		self.wssim = args.wssim
		self.ssim = env_utils.SSIM(window_size=5)
		self.wl2 = args.wl2
		self.wconsis = args.wconsis
		self.epslen = args.epslen
		#todo gpu set
		if args.gpu == -1:
			self.gpu = 'cpu'
		else:
			self.gpu = args.gpu
		#todo load models
		if args.ae =='vae_gan':
			self.ae = vae_gan.VAE(args.dim_s)
			self.dis = vae_wgan_gp.Discriminator()
		elif args.ae == 'vae_wgan':
			self.ae = vae_wgan.VAE(args.dim_s)
			self.dis = vae_wgan_gp.Discriminator().to(self.gpu)
		elif args.ae == 'vae_wgan_gp':
			self.ae = vae_wgan_gp.VAE(args.dim_s,config='vae')
			self.dis = vae_wgan_gp.Discriminator().to(self.gpu)
		elif args.ae == 'ae':
			self.ae  = ae.AE(args.dim_s)
			self.dis = vae_wgan_gp.Discriminator().to(self.gpu)
		elif args.ae =='vae':
			self.ae = vae_wgan_gp.VAE(args.dim_s)
			self.dis = vae_gan.Discriminator().to(self.gpu)
		else:
			raise ValueError("model:{} should be one of vae_gan,vae_wgan,ae".format(args.ae))
		#todo -------------------------------------init discriminator--------------------------------------------------
		self.dis = vae_wgan_gp.Discriminator().to(self.gpu)
		self.opt_dis = optim.Adam(params=self.dis.parameters(), lr=args.lr)


		try:
			self.ae.load_state_dict(torch.load(os.path.join(args.model_root,args.ae+str(args.dim_s)+'.pkl')))
			self.ae.to(self.gpu).eval()
			utils.freeze(self.ae)

		# self.dis.load_state_dict(torch.load(os.path.join(args.model_root, args.ae + '_dis.pkl')))
		# self.dis.to(self.gpu).eval()
		# utils.freeze(self.dis)
		except:
			raise ValueError("parameter of model:{} is not found in :{}".format(args.ae, args.model_root))

		#todo init discriminator


		#todo init dataset
		dset = datasets.IIPDataset(args.dst, args.dst_idx, crop_ratio=args.dst_cp)
		self.dldr = DataLoader(dset, batch_size=args.dst_bs, shuffle=True, num_workers=16, drop_last=True)
		print('data :{}  index:{}   crop:{}     bs:{}'.format(args.dst,args.dst_idx,args.dst_cp,args.bs))
		self.bs = self.dldr.batch_size

		#todo init spaces
		self.obsevation_space = args.dim_s
		self.action_space = args.dim_a
		self.max_a = args.max_a

		self.reset()
		# self.mean_rec_r,self.mean_consis_r,self.mean_l2_r = utils.average_scale(0.1),utils.average_scale(0.1),utils.average_scale(0.1)
		# self.half_rec_r,self.half_consis_r,self.half_l2_r = utils.average_scale(.1),utils.average_scale(.1),utils.average_scale(.1)

	def reset(self):
		try:
			self.true,self.img,self.loc = next(self.dst)

			self.img = self.img.to(self.gpu)
			self.true = self.true.to(self.gpu)
			self.loc = self.loc.to(self.gpu)

			self.s_true = self.ae.encode(self.true)
			self.s_noise = self.ae.encode(self.img)
			self.s = self.s_noise.clone()
			self.img_rec = self.ae.decode(self.s_noise)

			self.e = self.energy(self.true,self.img_rec,self.img_rec,self.s,self.s)
			self.num_step  = 0
		except:
			self.dst = iter(self.dldr)

			self.true, self.img, loc = next(self.dst)

			self.img = self.img.to(self.gpu)
			self.true = self.true.to(self.gpu)

			self.s_true = self.ae.encode(self.true)
			self.s = self.ae.encode(self.img)
			self.s_noise = self.s.clone()
			self.img_rec = self.ae.decode(self.s_noise)

			self.e = self.energy(self.true, self.img_rec, self.img_rec, self.s_noise,self.s)
			self.num_step = 0

	def train_dis(self):
		# todo train discriminator
		judge_real, judge_fake = self.dis(self.s_true), self.dis(self.s)
		# gp = vae_wgan_gp.gradient_penalty(self.dis, self.true, self.img_fake)
		l_dis = lgan.discriminator_loss(judge_real, judge_fake)
		# print('l_dis:{}'.format(l_dis))
		self.opt_dis.zero_grad()
		l_dis.backward()
		# if l_dis>=0.01:
		self.opt_dis.step()

	def observe(self):
		s = torch.cat([self.s_noise.view(self.bs,-1), self.s.view(self.bs,-1),self.loc(self.bs,-1)], dim=1).detach().cpu().numpy()
		return s

	def step(self,a):
		#todo ----------------------s-------------------------------
		state = self.observe()
		#todo ----------------------forward-------------------------
		self.s_fake = a+self.s
		self.img_fake = self.ae.decode(self.s_fake)

		# todo ----------------------update------------------------------
		# self.next_s()
		e = self.energy(self.true,self.img_fake,self.img_rec,self.s_noise,self.s_fake)
		self.s = self.s_fake.detach()
		self.a = a.detach().cpu()
		self.r = e-self.e
		self.e = e

		self.num_step+=1

		self.train_dis()

		self.buffer = (self.true.detach(), self.img.detach(), self.img_rec.detach(), self.img_fake.detach())
		# todo ----------------------terminal------------------------------
		if self.num_step<self.epslen:
			terminal = np.array([False] * self.bs).reshape(self.bs, 1)
		else:
			terminal = np.array([True] * self.bs).reshape(self.bs, 1)
		return state, a.detach().cpu().view(self.bs, -1).numpy(), self.r, self.observe(), terminal

	def energy(self,true,fake,rec,f_noise,f_now):
		#todo ----------------------reward--------------------------
		mse_r = -(true.view(self.dldr.batch_size,-1)-fake.view(self.dldr.batch_size,-1)).norm(2,dim=1,keepdim=True)\
				+(true.view(self.dldr.batch_size,-1)-rec.view(self.dldr.batch_size,-1)).norm(2,dim=1,keepdim=True)
		# l2_r = -(f_true-f_fake).norm(2,dim=1,keepdim=True)
		psnr_r = env_utils.psnr(true.view(self.dldr.batch_size, -1),fake.view(self.dldr.batch_size, -1)) - \
		         env_utils.psnr(true.view(self.dldr.batch_size, -1), rec.view(self.dldr.batch_size, -1))

		# ssim_r
		ssim_r = self.ssim(true, fake) - self.ssim(true, rec)
		#action norm reward
		l2_r = -((f_noise-f_now)**2).mean(dim=1,keepdim=True)
		#consis r
		consis_r = self.dis(f_now)

		#type change
		mse_r = mse_r.detach().cpu().numpy()
		psnr_r = psnr_r.detach().cpu().numpy()
		ssim_r = ssim_r.detach().cpu().numpy().reshape(self.dldr.batch_size,1)
		l2_r = l2_r.detach().cpu().numpy()
		consis_r = consis_r.detach().cpu().numpy()
		# print('mse_r:{}   psnr_r:{}     ssim_r:{}       l2_r:{}  consis_r:{}'.format(mse_r.mean(),psnr_r.mean(),ssim_r.mean(),l2_r.mean(),consis_r[0]))
		r = self.wmse*mse_r+self.wpsnr*psnr_r+self.wssim*ssim_r+self.wl2*l2_r+self.wconsis*consis_r
		return r





