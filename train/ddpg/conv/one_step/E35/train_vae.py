#todo"""-----------------import normal package-------------------------"""
import torch,torchvision,os,json,argparse,utils,tqdm
from torch.utils.data import DataLoader
from tqdm import trange
from torch import optim,nn
from tensorboardX import SummaryWriter

#todo"""--------------------------------import user package------------------------------------"""
from datasets import datasets,celeba
from train.ddpg.conv.one_step.E35 import tool

#todo"""--------------------------------args------------------------------------"""
parser = argparse.ArgumentParser(description='train ae')

log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')

parser.add_argument('--gpu',type=int,default=0,help='gpu id')
parser.add_argument('--multi-gpu',type=list,default=[0,1,2,3],help='multi gpu id')

parser.add_argument('--seed',type=int,default=0,help='random seed')
parser.add_argument('--dst',type=str,default='celeba',help='chose from [celeba,dataset]')
parser.add_argument('--img-size',type=int,default=256,help='chose from [celeba,dataset]')

train.add_argument('--epoch',type=int,default=15,metavar='TS',help='training steps')
train.add_argument('--bs',type=int,default=64,metavar='BS',help='training batch size')
train.add_argument('--crop',type=tuple,default=(0.4,0.4),help='crop ratio')
train.add_argument('--lr',type=float,default=1e-3,help=' learning rate')

log.add_argument('--log-dir',type=str,default='/home/pikey/Data/II/35/ae',help='log directory')

args = parser.parse_args()

# save args
config = vars(args)
if not os.path.exists(args.log_dir):
	os.makedirs(args.log_dir)
with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
	json.dump(config, f, cls=utils.DataEnc, indent=2)

#todo"""--------------------------------gpu preparation------------------------------------"""
if args.gpu == -1:
	gpu = 'cpu'
else:
	gpu = args.gpu
print('gpu :{}'.format(gpu))

#todo"""--------------------------------datapreparation------------------------------------"""
if args.dst == 'celeba':
	dset = celeba.D1('/home/pikey/DataSet/celeba','train',img_size=(args.img_size,args.img_size),crop_ratio=args.crop)
else:
	dset = datasets.IIPDataset('/home/pikey/DataSet/celeba','train',crop_ratio=args.crop)
dldr = DataLoader(dset,batch_size=args.bs,shuffle=True,num_workers=16,drop_last=True)
print('data :{}'.format(args.dst))

#todo --------------------------------modelpreparation------------------------------------
torch.manual_seed(args.seed)
m = tool.load_ae_model()
m.load_state_dict(torch.load('/home/pikey/Data/II/34/ae/model/epoch14.pkl'))
m = m.to(gpu)
if len(args.multi_gpu)!=1:
	m = nn.DataParallel(m,device_ids=args.multi_gpu)
opt = optim.Adam(m.parameters(),lr=args.lr)
print('model done')
print(m)
#todo --------------------------------logger init-------------------------------------
if os.path.exists(os.path.join(args.log_dir, 'train_result', 'img')) == False:
	os.makedirs(os.path.join(args.log_dir, 'train_result', 'img'))
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train_result'))
print('logger dir:{}'.format(os.path.join(args.log_dir, 'train_result', 'img')))

#todo"""--------------------------------train------------------------------------"""
def train():
	print('train begin')

	if not os.path.exists(os.path.join(args.log_dir, 'model')):
		os.makedirs(os.path.join(args.log_dir, 'model'))
	torch.save(m.module.cpu().state_dict(), os.path.join(args.log_dir, 'model', 'epoch{}.pkl'.format(0)))
	m.module.to(gpu)

	# train
	step = 0
	for epoch in trange(args.epoch):
		bt = 0
		for true, flaw, mask,patch,region in tqdm.tqdm(dldr,desc='Epoch:{}Batch'.format(epoch)):
			bt += 1
			step += 1
			#todo"""--------------------------------forward------------------------------------"""
			pos = torch.cat([true,torch.zeros_like(mask)],dim=1)
			neg = torch.cat([flaw,mask],dim=1)
			input=  torch.cat([neg,pos],dim=0).to(gpu)
			rec= m(input)
			loss = (rec-true.repeat(2,1,1,1).to(gpu)).abs().view(args.bs,-1).sum(1).mean()
			#todo"""--------------------------------train------------------------------------"""
			loss = loss.mean()
			opt.zero_grad()
			loss.backward()
			opt.step()

			#todo"""--------------------------------savedata------------------------------------"""
			# log
			print('loss_rec:{}'.format(loss))
			writer.add_scalar('loss',loss.detach())
			if step % 200 == 0:

				img = torchvision.utils.make_grid(
					[true[0].cpu(),
					 flaw[0].cpu(),
					 rec[args.bs].detach().cpu(),
					 rec[0].detach().cpu(),
					 ],

					nrow=4)
				img = utils.rescale(img)
				# writer.add_image('results', img, step)
				torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img',
															   'epoch{}_batch{}.png'.format(epoch, bt)))
		if (epoch+1)%5==0:
			if not os.path.exists(os.path.join(args.log_dir, 'model')):
				os.makedirs(os.path.join(args.log_dir, 'model'))
			torch.save(m.module.cpu().state_dict(), os.path.join(args.log_dir, 'model', 'epoch{}.pkl'.format(epoch+15)))
			m.module.to(gpu)

if __name__ == '__main__':
	train()