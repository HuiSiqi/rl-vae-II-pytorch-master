#todo"""-----------------import normal package-------------------------"""
import torch,torchvision,os,json,utils,setproctitle,pylib,torchlib,torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange,tqdm
from torch import optim
from tensorboardX import SummaryWriter
from torch import nn
#todo"""--------------------------------import user package------------------------------------"""
from ae import ae
import wgan_gp.model as wgan_gp
# from datasets import datasets
from datasets import celeba
# from vae_wgan import vae_wgan
#todo"""--------------------------------args------------------------------------"""

pylib.arg('--gpu',type=int,default=1,help='gpu id')
pylib.arg('--multi-gpu',type=list,default=[1,2,3],help='gpu id')
pylib.arg('--seed',type=int,default=0,help='random seed')

pylib.arg('--epoch',type=int,default=30,metavar='TS',help='training steps')
pylib.arg('--bs',type=int,default=16,metavar='BS',help='training batch size')
pylib.arg('--lr',type=float,default=2e-4,metavar='lr',help='learning rate')
pylib.arg('--sigma',type=float,default=10,metavar='Sigma',help='seed variance')
pylib.arg('--gamma',type=float,default=10,metavar='Gamma',help='weight of dis loss')
pylib.arg('--beta',type=float,default=0.4,help='Adam parameter')

pylib.arg('--log-dir',type=str,default='/home/pikey/Data/II/wgan/gp',help='log directory')
pylib.arg('--id',type=str,default='23',help='experiment id')

# pylib.arg('--z-dim',type=int,default=8,metavar='dz',help='channel of generated featuremap')
pylib.arg('--g-dim',type=int,default=256,metavar='dg',help='latent_space_dimension')
pylib.arg('--s-dim',type=int,default=3,metavar='ds',help='dimension of generated sample')
pylib.arg('--n-conv',type=int,default=6,help='down or upsample times')

args = pylib.args()
args.log_dir = os.path.join(args.log_dir,args.id)
utils.mkdir(args.log_dir)
#todo save args
pylib.args_to_yaml(args.log_dir,args)


#todo"""--------------------------------gpu preparation------------------------------------"""
if args.gpu == -1:
	gpu = 'cpu'
else:
	gpu = args.gpu
print('gpu :{}'.format(gpu))

#todo"""--------------------------------datapreparation------------------------------------"""
dset = celeba.IIPDataset('/home/pikey/DataSet/celeba','train',crop_ratio=(0,0))
dldr = DataLoader(dset,batch_size=args.bs,shuffle=True,num_workers=16,drop_last=True)
print('data :{}'.format('celeba,train'))

#todo --------------------------------modelpreparation------------------------------------
torch.manual_seed(args.seed)
m = wgan_gp.WGAN_GP(args.g_dim,dim_sample=args.s_dim,n_conv=args.n_conv,gamma=args.gamma).to(gpu)
m = nn.DataParallel(m,device_ids=args.multi_gpu)
gen_opt = optim.Adam(m.module.generator.parameters(),lr=args.lr,betas=(args.beta,0.999))
dis_opt = optim.Adam(m.module.discriminator.parameters(),lr=args.lr,betas=(args.beta,0.999))
# gen_opt = nn.DataParallel(gen_opt,device_ids=args.multi_gpu)
# dis_opt = nn.DataParallel(dis_opt,device_ids=args.multi_gpu)
print('model done')

#todo --------------------------------logger init-------------------------------------
if os.path.exists(os.path.join(args.log_dir, 'train_result','img')) == False:
	os.makedirs(os.path.join(args.log_dir, 'train_result', 'img'))
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train_result'))
print('logger dir:{}'.format(os.path.join(args.log_dir, 'train_result', 'img')))

#todo ----------------------------------set title-----------------------------------
setproctitle.setproctitle('wgan_gp gpu:{}'.format(args.gpu))

#todo -----------------------------------fix test seed ----------------------------
fix_seed = torch.randn(64,args.g_dim).to(gpu)
#todo"""--------------------------------train------------------------------------"""
def train():
	print('train begin')
	# sigma= utils.decay_scale(1e-2,1e-8,10000)
	# sigma = 0.3
	# train
	step = 0
	for epoch in trange(args.epoch,desc='Epoch'):
		bt = 0
		for true, flaw, region in tqdm(dldr,desc='Inner Loop'):
			bt += 1
			step += 1
			#todo"""--------------------------------forward------------------------------------"""
			true = true.to(gpu)

			seed =  torch.randn(args.bs,args.g_dim).to(gpu)

			# rec = AE(true)
			# true_f = AE.encode(true)
			gen_loss,dis_loss,fake = m((seed,true))
			gen_loss = gen_loss.mean()
			dis_loss = dis_loss.mean()
			gen_opt.zero_grad()
			gen_loss.backward(retain_graph=True)
			gen_opt.step()

			dis_opt.zero_grad()
			# dis_loss*=args.gamma
			dis_loss.backward()
			dis_opt.step()
			#todo"""--------------------------------savedata------------------------------------"""
			# log
			print('loss_gen:{},     loss_dis:{}'.format(gen_loss.detach().cpu().numpy(),dis_loss.detach().cpu().numpy()))
			# print('grad_gen:{},     grad_dis:{}'.format(utils.gradient_norm(m.module.generator),utils.gradient_norm(m.module.discriminator)))
			writer.add_scalar('loss_gen',gen_loss.mean(),step)
			writer.add_scalar('loss_dis',dis_loss.mean(),step)
			if step % 100 == 0:
				img = test()
				torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img',
				                                               'epoch{}_batch{}.png'.format(epoch, bt)))
		if not os.path.exists(os.path.join(args.log_dir, 'model')):
			os.makedirs(os.path.join(args.log_dir, 'model'))
		torch.save(m.state_dict(), os.path.join(args.log_dir, 'model', 'epoch{}.pkl'.format(epoch)))

def test():
	fake_img = m.module.generator(fix_seed.view(64,args.g_dim,1,1)).detach()
	img = torchvision.utils.make_grid(
					fake_img,
					nrow=8)
	img = utils.rescale(img)
	img = F.interpolate(img.unsqueeze(0),(1024,1024)).squeeze(0)
	return img

if __name__ == '__main__':
	train()