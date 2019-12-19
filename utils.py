import torch,os,json,types,imageio
from matplotlib import pyplot as plt
from consolecolor import FontColor,Colors

def load_model(m,dir,filename):
	dicr_dir = os.path.join(dir,'model',filename)
	m.load_state_dict(torch.load(dicr_dir))

class DataEnc(json.JSONEncoder):
	def default(self, o):
		if isinstance(o,types.FunctionType):
			return o.__name__

def save_img(img,dir,file):
	if not os.path.exists(dir):
		os.makedirs(dir)
	plt.imsave(os.path.join(dir,file), img.copy().squeeze())

def save_fig(fig,dir,file):
	if not os.path.exists(dir):
		os.makedirs(dir)
	fig.savefig(os.path.join(dir,file))

def img2gif(imgdir,gifname):
	images = []
	filenames = sorted((os.path.join(imgdir,fn) for fn in os.listdir(imgdir)
	                    if fn.endswith('.png')))
	for filename in filenames:
		images.append(imageio.imread(filename))
	imageio.mimsave(os.path.join(imgdir,gifname+'.gif'),images,duration=0.3)

class decay_scale():
	def __init__(self,begin,end,step):
		self.n = begin
		self.begin = begin
		self.end = end
		self.lr = (end-begin)/step

	def step(self):
		if abs(self.n-self.end)<1e-8:
			return
		else:
			self.n+=self.lr

	def __call__(self,):
		return self.n

def save_model(m,save_dir,name):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	torch.save(m.state_dict(), os.path.join(save_dir, name))

def gradient_norm(model):
	s = 0
	for p in model.parameters():
		try:
			param_norm = p.grad.data.norm(2)
			s += param_norm.item() ** 2
		except:
			pass
	print(FontColor.set_color('{}'.format(s**(1/2)), Colors.red))
	return s

def freeze(model):
	for p in model.parameters():
		p.requires_grad_(False)

def unfreeze(model):
	for p in model.parameters():
		p.requires_grad_(True)

def soft_update(target,source,tau):
	for target_param ,param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)