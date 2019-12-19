from torch import nn
import model.utils as model_utils
import torch

class Actor(nn.Module):
	def __init__(self,max_a):
		super(Actor, self).__init__()
		self.max_a = max_a
		# self.state_enc = model_utils.Model(
		# 	8,
		# 	config=[
		# 		'1K5D1S1C64B1',
		#         '1K3D1S1C128B1',
		#         '1K3D1S1C128B1',
		# 	        ]
		# )
		self.bbox_enc = model_utils.Model(
			4,
			config=[
				'0K4D1S0C8B0',
				'0K4D1S2C8B0',
				'0K4D1S2C8B0',
				'0K4D1S2C8B0',
			]
		)
		self.mix = model_utils.Model(
			24,
			config=
			[
				'1K3D1S1C64B0',
				'1K3D1S1C128B0',
				'1K3D1S1C256B0',
				'1K3D2S1C256B0',
				'1K3D4S1C256B0',
				'1K3D8S1C256B0',
				'1K3D16S1C256B0',
				'1K3D1S1C64B0',
				nn.Conv2d(64,8,3,1,1),
				nn.Tanh()
			 ]
		)
	def forward(self, state):
		f,bbox = state[:,:-4],state[:,-4:]
		bs,dim = f.size()
		f_noise,f_now = f.split(int(dim/2),dim=1)
		f_noise,f_now,bbox = f_noise.view(bs,8,32,32),f_now.view(bs,8,32,32),bbox.view(bs,4,1,1)

		# f_noise = self.state_enc(f_noise)
		# f_now = self.state_enc(f_now)
		f_bbox = self.bbox_enc(bbox)

		_ = torch.cat([f_noise,f_now,f_bbox],dim=1)
		action = self.mix(_)
		return action*self.max_a

class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()
		# self.state_enc = model_utils.Model(
		# 	128,
		# 	config=['1K5D1S1C128B0',
		# 	        '1K3D1S2C256B0',
		# 	        '1K3D1S2C256B0',
		# 	        '1K3D1S1C256B0',
		# 	        ]
		# )
		self.bbox_enc = model_utils.Model(
			4,
			config=[
				'0K4D1S0C8B0',
				'0K4D1S2C8B0',
				'0K4D1S2C8B0',
				'0K4D1S2C8B0',
			]
		)
		# self.action_enc = model_utils.Model(
		# 	128,
		# 	config=['1K5D1S1C128B1',
		# 	        '1K3D1S2C256B1',
		# 	        '1K3D1S2C256B1',
		# 	        '1K3D1S1C256B1',
		# 	        ]
		# )
		self.mix = model_utils.Model(
			8*4,
			config=
			[
				'1K3D1S1C64B0',
				'1K3D1S1C256B0',
				'1K3D2S1C256B0',
				'1K3D4S1C256B0',
				'1K3D8S1C256B0',
				'1K3D16S1C256B0',
				'1K3D1S2C512B0',
				'1K3D1S2C512B0',
				'1K3D1S2C1024B0',
				'1K4D1S0C1024B0',
				nn.Conv2d(1024,1,1,1),
			]
		)
	def forward(self, s,a):
		f, bbox = s[:, :-4], s[:, -4:]
		bs, dim = f.size()
		f_noise, f_now = f.split(int(dim / 2), dim=1)
		f_noise, f_now, bbox = f_noise.view(bs, 8, 32, 32), f_now.view(bs, 8, 32, 32), bbox.view(bs, 4, 1, 1)
		a = a.view(bs,8,32,32)

		# f_noise = self.state_enc(f_noise)
		# f_now = self.state_enc(f_now)
		# f_a = self.action_enc(a)
		f_bbox = self.bbox_enc(bbox)

		_ = torch.cat([f_noise,f_now,f_bbox,a],dim=1)
		value = self.mix(_)
		return value