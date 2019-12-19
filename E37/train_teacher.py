#todo"""-----------------import normal package-------------------------"""
import sys
sys.path.append('/home/pikey/PycharmProjects/rl-vae-II-pytorch-master')
import torchvision,os, utils, tqdm
from tensorboardX import SummaryWriter
from tqdm import trange
from torch import nn,optim
from datasets import celeba
from torch.utils.data import DataLoader
#todo------------------------------import user package-------------------------------
import pylib
from E36 import tool

#todo get path
dirname,filename = os.path.split(os.path.abspath(__file__))

pylib.arg('--gpu',type=int,default=0)
pylib.arg('--multi-gpu',type=tuple,default=[0,1,2,3])
pylib.arg('--seed',type=int,default=2020)
pylib.arg('--log-dir',type=str,default='/home/pikey/Data/II/36/regression/',help='log directory')
#todo -----------------------train config ----------------------------------------
pylib.arg('--epoch',type=int,default=40,help='train steps')

pylib.arg('--lr',type=float,default=1e-3,help='learning rate')
# pylib.arg('--max-a',type=int,default=20,help='batch size')

#todo ---------------------------env config--------------------------------
pylib.arg('--dst',default='celeba',choices=['dataset','celeba'])
pylib.arg('--dst-cp',type=tuple,default=(0.4,0.4))
pylib.arg('--dst-bs',type=int,default=128,help='dataset batch size')
pylib.arg('--dst-nwks',type=int,default=16,help='dataset number workers')

# pylib.arg('--ae-dir',type=str,default='/home/pikey/Data/II/35/ae/model/epoch14.pkl',help='log directory')

pylib.arg('--feature-state',type=bool,default=False,help='dataset batch size')

pylib.arg('--img-size',type=int,default=256,help='dataset batch size')


#todo ----------------------agent config------------------------------------

#todo actor config
pylib.arg('--state-size',type=tuple,default=(8,32,32),help='(channel,h,w)')
pylib.arg('--action-size',type=tuple,default=(8,32,32),help='(channel,h,w)')

#todo -------------------------------init----------------------------------------
#todo args
args = pylib.args()
#todo mkdirs
utils.mkdir(os.path.join(args.log_dir, 'train_result', 'img'))
utils.mkdir(os.path.join(args.log_dir))
#todo gpu
if args.gpu == -1:
	gpu = 'cpu'
else:
	gpu = args.gpu
print('gpu :{}'.format(gpu))
multi_gpu = len(args.multi_gpu)>1
# args.log_dir+=filename.split('.')[0]
pylib.args_to_yaml(args.log_dir+'/config',args)
#todo writter
writer = SummaryWriter()
#todo ---------------------------------------------------dataset--------------------------------------
if args.dst == 'celeba':
	dst = celeba.D1('/home/pikey/DataSet/celeba', 'train', crop_ratio=args.dst_cp,
	                     img_size=(args.img_size, args.img_size))
	dldr = DataLoader(dst, args.dst_bs, shuffle=True, drop_last=True, num_workers=args.dst_nwks)

#todo model
player = tool.load_regressor()
player.to(args.gpu)
if multi_gpu:
	player = nn.DataParallel(player,device_ids=args.multi_gpu)
opt = optim.Adam(player.parameters(),lr=args.lr)

# todo -----------------------------train--------------------------------------
step = 0
def train():
	global step
	for i in trange(int(args.epoch),desc='Epoch'):
		for true, img, mask, true_patch, bbox in tqdm.tqdm(dldr,desc='Step'):

			true,img = true.to(gpu),img.to(gpu)
			out = player(img)
			loss = (out-true).abs().view(args.dst_bs,-1).sum(dim=1).mean()
			print('loss:{}'.format(loss))
			opt.zero_grad()
			loss.backward()
			opt.step()
			writer.add_scalar('loss',loss,global_step=i)
			if step%200==0:
				log(true, out, mask)

			step += 1
def log(true, out, mask):
	# todo agent
	patch_mask = mask.repeat(1, 3, 1, 1).clone().to(gpu)
	mix = true*(1-patch_mask)+patch_mask*out

	true, noisy, mix, inpainted = true.clone(),true*(1-patch_mask)-patch_mask,mix, out.clone()
	img = torchvision.utils.make_grid(
		[true[0].cpu(), noisy[0].cpu(),mix[0].cpu(), inpainted[0].cpu()], nrow=4)
	img = utils.rescale(img)  # change image to 0-1
	# writer.add_image('results', img, step)
	torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img',
	                                               'step:{}.png'.format(step)))

	if step%2000==0:
		if multi_gpu:
			utils.save_model(player.module, args.log_dir, 'model.pth')
		else:
			utils.save_model(player, args.log_dir, 'model.pth')
		player.to(args.gpu)

if __name__ == '__main__':
	print(player)
	train()





