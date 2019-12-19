import model.utils as model_utils,torch
from torch import nn
import ae
from environment import envs
from DDPG import agent
def load_ae_model():
	enc = model_utils.Model(4,
	                        [
		                        '1K5D1S1C32B1',
		                        '1K3D1S2C64B1',
		                        '1K3D1S1C64B1',
		                        '1K3D1S2C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D1S2C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D1S2C128B1',
		                        '1K3D2S1C128B1',
		                        '1K3D4S1C128B1',
		                        '1K3D8S1C128B1',
		                        '1K3D16S1C128B1',
		                        '1K3D1S1C64B1',
		                        '1K3D1S1C8B1',
		                    ]
	                        )

	dec = model_utils.Model(8,
	                [
		                '0K4D1S2C256B1',
		                '0K4D1S2C128B1',
		                '0K4D1S2C64B1',
		                '0K4D1S2C3B1',
		                nn.Tanh()
	                ])

	m = ae.model.AE(enc,dec)
	return m

def load_env(args):
	ae = load_ae_model()
	ae.load_state_dict(torch.load(args.ae_dir))
	e = envs.E34(args,ae)
	return e

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
			]
		)
		self.mix = model_utils.Model(
			24,
			config=
			[
				'1K3D1S1C1024B0',
				'1K3D1S1C512B0',
				'1K3D2S1C256B0',
				'1K3D4S1C256B0',
				'1K3D8S1C256B0',
				'1K3D16S1C64B0',
				nn.Conv2d(64,8,3,1,1),
				nn.Tanh()
			 ]
		)
	def forward(self, state):
		f,bbox = state[:,:-4],state[:,-4:]
		bs,dim = f.size()
		f_noise,f_now = f.split(int(dim/2),dim=1)
		f_noise,f_now,bbox = f_noise.view(bs,8,16,16),f_now.view(bs,8,16,16),bbox.view(bs,4,1,1)

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
				'1K3D1S2C1024B0',
				'1K4D1S0C1024B0',
				nn.Conv2d(1024,1,1,1),
			]
		)
	def forward(self, s,a):
		f, bbox = s[:, :-4], s[:, -4:]
		bs, dim = f.size()
		f_noise, f_now = f.split(int(dim / 2), dim=1)
		f_noise, f_now, bbox = f_noise.view(bs, 8, 16, 16), f_now.view(bs, 8, 16, 16), bbox.view(bs, 4, 1, 1)
		a = a.view(bs,8,16,16)

		# f_noise = self.state_enc(f_noise)
		# f_now = self.state_enc(f_now)
		# f_a = self.action_enc(a)
		f_bbox = self.bbox_enc(bbox)

		_ = torch.cat([f_noise,f_now,f_bbox,a],dim=1)
		value = self.mix(_)
		return value

class LinearActor(nn.Module):
	def __init__(self,max_a):
		super(LinearActor, self).__init__()
		self.max_a = max_a
		# self.state_enc = model_utils.LinearModel(
		# 	8*16*16,
		# 	config=[
		# 		'C1024B1',
		# 		'C512B1',
		# 	    ]
		# )
		self.bbox_enc = model_utils.LinearModel(
			4,
			config=[
				'C64B0',
				'C512B0',
				'C2048B0',
			]
		)
		self.mix = model_utils.LinearModel(
			2048*3,
			config=
			[
				'C4096B0',
				nn.Linear(4096,2048),
				nn.Tanh()
			 ]
		)
	def forward(self, state):
		f,bbox = state[:,:-4],state[:,-4:]
		bs,dim = f.size()
		f_noise,f_now = f.split(int(dim/2),dim=1)
		f_noise,f_now,bbox = f_noise.view(bs,-1),f_now.view(bs,-1),bbox.view(bs,-1)

		# f_noise = self.state_enc(f_noise)
		# f_now = self.state_enc(f_now)
		f_bbox = self.bbox_enc(bbox)

		_ = torch.cat([f_noise,f_now,f_bbox],dim=1)
		action = self.mix(_)
		return action*self.max_a

class LinearCritic(nn.Module):
	def __init__(self):
		super(LinearCritic, self).__init__()
		# self.state_enc = model_utils.LinearModel(
		# 	8*16*16,
		# 	config=[
		# 		'C1024B1',
		# 		'C512B1',
		# 	        ]
		# )
		self.bbox_enc = model_utils.LinearModel(
			4,
			config=[
				'C64B0',
				'C512B0',
				'C2048B0',
			]
		)
		# self.action_enc = model_utils.LinearModel(
		# 	8 * 16 * 16,
		# 	config=[
		# 		'C1024B1',
		# 		'C512B1',
		# 	]
		# )
		self.mix = model_utils.LinearModel(
			2048*4,
			config=
			[
				'C6144B0',
				'C4096B0',
				'C2048B0',
				'C1024B0',
				nn.Linear(1024,1),
			]
		)
	def forward(self, s,a):
		f, bbox = s[:, :-4], s[:, -4:]
		bs, dim = f.size()
		f_noise, f_now = f.split(int(dim / 2), dim=1)
		f_noise, f_now, bbox = f_noise.view(bs, -1), f_now.view(bs, -1), bbox.view(bs,-1)
		f_a = a.view(bs,-1)

		# f_noise = self.state_enc(f_noise)
		# f_now = self.state_enc(f_now)
		# f_a = self.action_enc(f_a)
		f_bbox = self.bbox_enc(bbox)

		_ = torch.cat([f,f_bbox,f_a],dim=1)
		value = self.mix(_)
		return value

def load_agent(args):
	player = agent.DDPG(args,Actor,Critic)
	return player



