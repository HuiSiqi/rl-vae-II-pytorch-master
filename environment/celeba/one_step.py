#todo ------------------------------------------------------basic packadges-------------------------------------------
import torch,os,utils,numpy as np,torch.nn.functional as F
from datasets import celeba
from torch.utils.data import DataLoader
from torch import optim
from . import utils as env_utils
#todo -----------------------load model packages--------------------------
from vae_wgan import vae_wgan
from vae_wgan_gp import vae_wgan_gp
from vae_gan import vae_gan
from ae import ae
from lgan import lgan
from vae_wgan_gp import vae_wgan_gp
import cv2

class env():
	def __init__(self,args):
		#todo init reward weight
		self.wmse = args.wmse
		self.wpsnr = args.wpsnr
		self.wssim = args.wssim
		self.wl2 = args.wl2
		self.wconsis = args.wconsis

		self.ssim = env_utils.SSIM(window_size=20)
		#todo gpu set
		if args.gpu == -1:
			self.gpu = 'cpu'
		else:
			self.gpu = args.gpu
		#todo load models
		if args.ae =='vae_gan':
			self.ae = vae_gan.VAE(args.dim_s,'conv')
			self.dis = vae_wgan_gp.Discriminator()
		elif args.ae == 'vae_wgan':
			self.ae = vae_wgan.VAE(args.dim_s,'conv')
			self.dis = vae_wgan_gp.Discriminator().to(self.gpu)
		elif args.ae == 'vae_wgan_gp':
			self.ae = vae_wgan_gp.VAE(args.dim_s,config='conv')
			self.dis = vae_wgan_gp.Discriminator().to(self.gpu)
		elif args.ae == 'ae':
			self.ae  = ae.AE(args.dim_s,'conv')
			self.dis = vae_wgan_gp.Discriminator().to(self.gpu)
		elif args.ae =='vae':
			self.ae = vae_wgan_gp.VAE(args.dim_s,'conv')
			self.dis = vae_gan.Discriminator().to(self.gpu)
		else:
			raise ValueError("model:{} should be one of vae_gan,vae_wgan,ae".format(args.ae))
		#todo ----------------------------------discriminator init----------------------------------
		self.dis = vae_wgan_gp.Discriminator().to(self.gpu)
		self.loc_dis = vae_wgan_gp.Discriminator().to(self.gpu)
		self.opt_dis = optim.Adam(params=self.dis.parameters(), lr=args.lr*0.5)
		self.opt_loc_dis = optim.Adam(params=self.loc_dis.parameters(), lr=args.lr*0.5)


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
			dset = celeba.IIPDataset_Random(args.dst,args.dst_idx,crop_range=args.dst_cp)
		else:
			dset = celeba.IIPDataset(args.dst, args.dst_idx, crop_ratio=args.dst_cp)
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
		self.dst = iter(self.dldr)
		try:
			self.true, self.img, self.loc = next(self.dst)
		except StopIteration:
			print(
				'the size of dataset is less than dst_cp:{} please reduce the args.dst_cp'.format(self.dldr.batch_size))
			raise StopIteration
		self.img = self.img.to(self.gpu)
		self.true = self.true.to(self.gpu)
		self.loc = self.loc.to(self.gpu)
		self.s = self.ae.encode(self.img)
		self.s_true = self.ae.encode(self.true)
		loc = self.loc.clone().int()
		self.x, self.dx, self.y, self.dy = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]
		self.patch_mask = torch.zeros_like(self.true)
		for i, (x, dx, y, dy) in enumerate(zip(self.x, self.dx, self.y, self.dy)):
			self.patch_mask[i, :, x:x + dx, y:y + dy] = 1
		self.patch_mask = self.patch_mask.byte()
		self.true_patch = self.true.masked_select(self.patch_mask).view(self.bs, 3, self.dx[0], self.dy[0])


	def next_s(self):
		self.train_dis()
		self.train_loc_dis()
		self.buffer = (self.true.detach(),self.img.detach(),self.img_rec.detach(),self.img_fake.detach(),self.img_mix.detach())
		try:
			self.true,self.img,self.loc = next(self.dst)
		except StopIteration:
			self.dst = iter(self.dldr)
			self.true, self.img, self.loc = next(self.dst)
		self.img = self.img.to(self.gpu)
		self.true = self.true.to(self.gpu)
		self.loc = self.loc.to(self.gpu)
		self.s = self.ae.encode(self.img)
		self.s_true = self.ae.encode(self.true)

		loc = self.loc.clone().int()
		self.x, self.dx, self.y, self.dy = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]
		self.patch_mask = torch.zeros_like(self.true)
		for i, (x, dx, y, dy) in enumerate(zip(self.x, self.dx, self.y, self.dy)):
			self.patch_mask[i, :, x:x + dx, y:y + dy] = 1
		self.patch_mask = self.patch_mask.byte()
		self.true_patch = self.true.masked_select(self.patch_mask).view(self.bs, 3, self.dx[0], self.dy[0])


	def train_dis(self):
		# todo train discriminator
		true = F.interpolate(self.true,(64,64)).detach()
		false = F.interpolate(self.img_fake,(64,64)).detach()

		judge_real, judge_fake = self.dis(true), self.dis(false)
		gp = vae_wgan_gp.gradient_penalty(self.dis, self.true, self.img_fake)
		l_dis = vae_wgan_gp.discriminator_loss(judge_real, judge_fake)+gp
		print('l_dis:{}'.format(l_dis))
		self.opt_dis.zero_grad()
		l_dis.backward()
		self.opt_dis.step()

	def train_loc_dis(self):
		# todo train discriminator
		true_patch = F.interpolate(self.true_patch,(64,64),mode='bilinear',align_corners=True).detach()
		fake_patch = F.interpolate(self.fake_patch,(64,64),mode='bilinear',align_corners=True).detach()

		judge_real, judge_fake = self.loc_dis(true_patch), self.loc_dis(fake_patch)
		gp = vae_wgan_gp.gradient_penalty(self.loc_dis, true_patch, fake_patch)
		l_dis = vae_wgan_gp.discriminator_loss(judge_real, judge_fake)+gp
		# print('l_dis:{}'.format(loss))
		self.opt_loc_dis.zero_grad()
		l_dis.backward()
		self.opt_loc_dis.step()

	def observe(self):
		s = torch.cat([self.s,self.s,self.loc],dim=1)
		s = s.cpu().view(self.bs, -1).numpy()
		return s

	def step(self,a):
		#todo ----------------------s-------------------------------
		s = self.observe()

		#todo ----------------------forward-------------------------
		self.s_fake = a+self.s
		self.img_rec = self.ae.decode(self.s)
		self.img_fake = self.ae.decode(self.s_fake)
		self.fake_patch = self.img_fake.masked_select(self.patch_mask).view(self.bs, 3, self.dx[0], self.dy[0])
		#todo create mixed img
		add = self.patch_mask.float()*self.img_fake+self.patch_mask.float()
		self.img_mix = self.img.clone()+add.clone()
		cv2.waitKey(0)
		#todo ----------------------reward--------------------------
		#mse_r
		mse_r = ((self.true.view(self.dldr.batch_size,-1)-self.img_rec.view(self.dldr.batch_size,-1))**2).mean(1,keepdim=True)\
			-((self.true.view(self.dldr.batch_size,-1)-self.img_fake.view(self.dldr.batch_size,-1))**2).mean(1,keepdim=True)
		#psnr_r
		psnr_r = env_utils.psnr(self.true.view(self.dldr.batch_size,-1),self.img_fake.view(self.dldr.batch_size,-1))-\
		         env_utils.psnr(self.true.view(self.dldr.batch_size,-1),self.img_rec.view(self.dldr.batch_size,-1))
		# ppsnr_r =  env_utils.ppsnr(self.true.view(self.dldr.batch_size,-1),self.img_fake.view(self.dldr.batch_size,-1))-\
		#          env_utils.ppsnr(self.true.view(self.dldr.batch_size,-1),self.img_rec.view(self.dldr.batch_size,-1))
		#ssim_r
		ssim_r = self.ssim(self.true,self.img_fake)-self.ssim(self.true,self.img_rec)
		#action_norm_r
		# l2_r =  -((a**2).mean(1,keepdim=True))/(self.max_a**2)
		l2_r =  -((a**2).mean(1,keepdim=True))
		#consistance_r
		# judge_r = self.lgan.dis(self.s_fake)
		img_fake = F.interpolate(self.img_fake, (64, 64))
		consistance_r = self.dis(img_fake).sigmoid()
		fake_patch = F.interpolate(self.fake_patch, (64, 64), mode='bilinear',align_corners=True)
		consistance_r += self.loc_dis(fake_patch).sigmoid()

		ssim_r = ssim_r.detach().cpu().numpy().reshape(self.dldr.batch_size,1)
		psnr_r = psnr_r.detach().cpu().numpy()
		# ppsnr_r  = ppsnr_r.detach().cpu().numpy()
		mse_r = mse_r.detach().cpu().numpy()
		l2_r = l2_r.detach().cpu().numpy()
		consistance_r = consistance_r.detach().cpu().numpy()
		print('mse_r:{}   psnr_r:{}     ssim_r:{}     l2_r:{}    consis_r:{}'.
		      format(self.wmse*mse_r.mean(),self.wpsnr*psnr_r.mean(),self.wssim*ssim_r.mean(),self.wl2*l2_r.mean(),self.wconsis*consistance_r.mean()))
		r = self.wmse*mse_r+self.wssim*ssim_r+self.wpsnr*(psnr_r)+self.wl2*l2_r+self.wconsis*consistance_r

		#todo ----------------------s_------------------------------
		self.next_s()
		s_ = np.zeros_like(s)
		#todo ----------------------terminal------------------------------
		terminal = np.array([True]*self.bs).reshape(self.bs,1)
		self.a = a.detach().cpu()

		return s,a.detach().cpu().view(self.bs,-1).numpy(),r,s_,terminal


