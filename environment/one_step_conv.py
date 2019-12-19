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
		self.wl2 = args.wl2
		self.wconsis = args.wconsis
		self.epslen = args.epslen

		self.ssim = env_utils.SSIM(window_size=5)
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
		if args.dst_random_noise:
			dset = datasets.IIPDataset_Random(args.dst,args.dst_idx,crop_range=args.dst_cp)
		else:
			dset = datasets.IIPDataset(args.dst, args.dst_idx, crop_ratio=args.dst_cp)
		self.dldr = DataLoader(dset, batch_size=args.dst_bs, shuffle=True, num_workers=16, drop_last=True)
		print('data :{}  index:{}   crop:{}     bs:{}'.format(args.dst,args.dst_idx,args.dst_cp,args.bs))
		self.bs = self.dldr.batch_size

		#todo init spaces
		self.obsevation_space = (self.bs,3,64,64)
		self.action_space = args.dim_a
		self.max_a = args.max_a

		self.reset()
		# self.mean_rec_r,self.mean_consis_r,self.mean_l2_r = utils.average_scale(0.1),utils.average_scale(0.1),utils.average_scale(0.1)
		# self.half_rec_r,self.half_consis_r,self.half_l2_r = utils.average_scale(.1),utils.average_scale(.1),utils.average_scale(.1)

	def reset(self):
		self.dst = iter(self.dldr)
		try:
			self.true, self.img, self.loc = next(self.dst)
			self.img = self.img.to(self.gpu)
			self.true = self.true.to(self.gpu)
			self.loc = self.loc.to(self.gpu)
			self.s = self.ae.encode(self.img)
			self.s_true = self.ae.encode(self.true)
			self.img_rec = self.ae.decode(self.s)
		except StopIteration:
			print(
				'the size of dataset is less than dst_cp:{} please reduce the args.dst_cp'.format(self.dldr.batch_size))
			raise StopIteration

	def next_s(self):
		self.train_dis()
		try:
			self.buffer = (self.true.detach(),self.img.detach(),self.img_rec.detach(),self.img_fake.detach())
			self.true,self.img,self.loc = next(self.dst)
			self.img = self.img.to(self.gpu)
			self.true = self.true.to(self.gpu)
			self.loc = self.loc.to(self.gpu)
			self.s = self.ae.encode(self.img)
			self.img_rec = self.ae.decode(self.s)
			self.s_true = self.ae.encode(self.true)
		except StopIteration:
			self.dst = iter(self.dldr)

	def train_dis(self):
		# todo train discriminator
		judge_real, judge_fake = self.dis(self.true), self.dis(self.img_fake)
		gp = vae_wgan_gp.gradient_penalty(self.dis, self.true, self.img_fake)
		l_dis = vae_wgan_gp.discriminator_loss(judge_real, judge_fake)+gp
		print('l_dis:{}'.format(l_dis))
		self.opt_dis.zero_grad()
		l_dis.backward()
		self.opt_dis.step()

	def observe(self):
		s = self.img.cpu().view(self.bs,-1)
		bbox = self.loc.cpu().view(self.bs,-1)
		ob = torch.cat([s,bbox],dim=1).numpy()
		return ob

	def step(self,a):
		#todo ----------------------s-------------------------------
		s = self.observe()
		#todo ----------------------forward-------------------------
		self.img_fake = self.ae.decode(a)
		#todo ----------------------reward--------------------------
		#mse_r
		mse_r = ((self.true.view(self.dldr.batch_size,-1)-self.img_rec.view(self.dldr.batch_size,-1))**2).mean(1,keepdim=True)\
			-((self.true.view(self.dldr.batch_size,-1)-self.img_fake.view(self.dldr.batch_size,-1))**2).mean(1,keepdim=True)
		#psnr_r
		psnr_r = env_utils.psnr(self.true.view(self.dldr.batch_size,-1),self.img_fake.view(self.dldr.batch_size,-1))-\
		         env_utils.psnr(self.true.view(self.dldr.batch_size,-1),self.img_rec.view(self.dldr.batch_size,-1))
		#ssim_r
		ssim_r = self.ssim(self.true,self.img_fake)-self.ssim(self.true,self.img_rec)
		#action_norm_r
		# l2_r =  -((a**2).mean(1,keepdim=True))/(self.max_a**2)
		l2_r =  -((a-self.s)**2).mean(1,keepdim=True)
		# l2_r = np.zeros_like(ssim_r)
		#consistance_r
		# judge_r = self.lgan.dis(self.s_fake)
		consistance_r = self.dis(self.img_fake).sigmoid()

		ssim_r = ssim_r.detach().cpu().numpy().reshape(self.dldr.batch_size,1)
		psnr_r = psnr_r.detach().cpu().numpy()
		mse_r = mse_r.detach().cpu().numpy()
		l2_r = l2_r.detach().cpu().numpy()
		consistance_r = consistance_r.detach().cpu().numpy()
		print('mse_r:{}   psnr_r:{}    ssim_r:{}     l2_r:{}    consis_r:{}'.
		      format(self.wmse*mse_r.mean(),self.wpsnr*psnr_r.mean(),self.wssim*ssim_r.mean(),self.wl2*l2_r.mean(),self.wconsis*consistance_r.mean()))
		r = self.wmse*mse_r+self.wssim*ssim_r+self.wpsnr*psnr_r+self.wl2*l2_r+self.wconsis*consistance_r

		#todo ----------------------s_------------------------------
		self.next_s()
		s_ = np.zeros_like(s)

		#todo ----------------------terminal------------------------------
		terminal = np.array([False]*self.bs).reshape(self.bs,1)
		self.a = a.detach().cpu()

		return s,self.a.view(self.bs,-1).numpy(),r,s_,terminal




