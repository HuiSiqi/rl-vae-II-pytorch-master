#todo ------------------------------------------------------basic packadges-------------------------------------------
import torch,os,utils,numpy as np
from datasets import datasets
from torch.utils.data import DataLoader

#todo -----------------------load model packages--------------------------
from vae_wgan import vae_wgan
from vae_wgan_gp import vae_wgan_gp
from vae_gan import vae_gan
from ae import ae
from lwgan import lwgan
from lwgan_gp import lwgan_gp
from lgan import lgan


class env():
	def __init__(self,args):
		#todo init reward weight
		self.wrec = args.wrec
		self.wl2 = args.wl2
		self.wjudge = args.wjudge

		#todo gpu set
		if args.gpu == -1:
			self.gpu = 'cpu'
		else:
			self.gpu = args.gpu
		#todo load models
		if args.ae =='vae_gan':
			self.ae = vae_gan.VAE(args.dim_s)
		elif args.ae == 'vae_wgan':
			self.ae = vae_wgan.VAE(args.dim_s)

		elif args.ae == 'vae_wgan_gp':
			self.ae = vae_wgan_gp.VAE(args.dim_s)
		elif args.ae == 'ae':
			self.ae  = ae.AE(args.dim_s)
		elif args.ae =='vae':
			self.ae = vae_gan.VAE(args.dim_s)
		else:
			raise ValueError("model:{} should be one of vae_gan,vae_wgan,ae".format(args.ae))

		if args.lgan =='lgan':
			self.lgan = lgan.LGAN(args.dim_a,args.dim_s)
		elif args.lgan == 'lwgan':
			self.lgan = lwgan.LWGAN(args.dim_a,args.dim_s)
		elif args.lgan == 'lwgan_gp':
			self.lgan = lwgan_gp.LWGAN_GP(args.dim_a,args.dim_s)
		else:
			raise ValueError("model:{} should be one of lgan,lwgan".format(args.lgan))

		try:
			self.ae.load_state_dict(torch.load(os.path.join(args.model_root,args.ae+'.pkl')))
			self.ae.to(self.gpu).eval()
			utils.freeze(self.ae)
		except:
			raise ValueError("parameter of model:{} is not found in :{}".format(args.ae, args.model_root))

		try:
			self.lgan.load_state_dict(torch.load(os.path.join(args.model_root,args.lgan+'.pkl')))
			self.lgan.to(self.gpu).eval()
			utils.freeze(self.lgan)
		except:
			raise ValueError("parameter of model:{} is not found in :{}".format(args.lgan, args.model_root))

		#todo init dataset
		dset = datasets.IIPDataset(args.dst, args.dst_idx, crop_ratio=args.dst_cp)
		self.dldr = DataLoader(dset, batch_size=args.dst_bs, shuffle=True, num_workers=16, drop_last=True)
		print('data :{}  index:{}   crop:{}     bs:{}'.format(args.dst,args.dst_idx,args.dst_cp,args.bs))
		self.bs = self.dldr.batch_size

	def reset(self):
		self.dst = iter(self.dldr)
		try:
			self.true,self.img,loc = next(self.dst)
			self.img = self.img.to(self.gpu)
			self.s = self.ae.encode(self.img)
		except StopIteration:
			print('the size of dataset is less than dst_cp:{} please reduce the args.dst_cp'.format(self.dldr.batch_size))
			raise StopIteration

	def next_s(self):
		try:
			self.buffer = (self.true.detach(),self.img.detach(),self.img_rec.detach(),self.img_fake.detach())
			self.true,self.img,loc = next(self.dst)
			self.img = self.img.to(self.gpu)
			self.s = self.ae.encode(self.img)

		except StopIteration:
			self.dst = iter(self.dldr)

	def observe(self):
		s = self.s.cpu().view(self.bs, -1).numpy()
		return s

	def step(self,a):
		#todo ----------------------s-------------------------------
		s = self.observe()
		#todo ----------------------forward-------------------------
		self.s_fake = self.lgan.gen(a)+self.s
		self.img_rec = self.ae.decode(self.s)
		self.img_fake = self.ae.decode(self.s_fake)
		#todo ----------------------reward--------------------------
		rec_r = -((self.true.view(self.dldr.batch_size,-1)-self.img_fake.detach().cpu().view(self.dldr.batch_size,-1))**2).mean(1,keepdim=True)
		l2_r = -(a**2).mean(1,keepdim=True)
		judge_r = self.lgan.dis(self.s_fake)
		rec_r = rec_r.detach().cpu().numpy()
		l2_r = l2_r.detach().cpu().numpy()
		judge_r = judge_r.detach().cpu().numpy()
		r = self.wrec*rec_r+self.wl2*l2_r+self.wjudge*judge_r
		#todo ----------------------s_------------------------------
		self.next_s()
		s_ = np.zeros_like(s)
		#todo ----------------------terminal------------------------------
		terminal = np.array([False]*self.bs).reshape(self.bs,1)
		self.a = a.detach().cpu()
		return s,a.detach().cpu().view(self.bs,-1).numpy(),r,s_,terminal



