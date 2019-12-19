from model import utils as model_utils
from torch import nn
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
				'0K4D1S0C8B1',
				'0K4D1S2C8B1',
				'0K4D1S2C8B1',
				'0K4D1S2C8B1',
			]
		)
		self.mix = model_utils.Model(
			24,
			config=
			[
				'1K3D1S2C64B1',
				'1K3D1S2C128B1',
				'1K3D1S1C256B1',
				'1K3D2S1C256B1',
				'1K3D4S1C256B1',
				'1K3D8S1C256B1',
				'0K4D1S2C256B1',
				'0K4D1S2C256B1',
				'1K3D1S1C64B1',
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


