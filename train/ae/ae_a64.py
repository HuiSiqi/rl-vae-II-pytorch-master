#todo"""-----------------import normal package-------------------------"""
import torch,torchvision,os,json,argparse,utils
from torch.utils.data import DataLoader
from tqdm import trange
from torch import optim
from tensorboardX import SummaryWriter

#todo"""--------------------------------import user package------------------------------------"""
from ae import ae
from datasets import datasets

#todo"""--------------------------------args------------------------------------"""
parser = argparse.ArgumentParser(description='train ae')

log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')

parser.add_argument('--gpu',type=int,default=0,help='gpu id')
parser.add_argument('--seed',type=int,default=0,help='random seed')

train.add_argument('--epoch',type=int,default=400,metavar='TS',help='training steps')
train.add_argument('--bs',type=int,default=64,metavar='BS',help='training batch size')
train.add_argument('--lr',type=float,default=1e-3,metavar='lr',help='learning rate')
train.add_argument('--loss',default=ae.compute_loss,metavar='LOSS',help='loss function')

log.add_argument('--log-dir',type=str,default='/home/pikey/Data/II/ae',help='log directory')

model.add_argument('--z-dim',type=int,default=1024,metavar='dz',help='latent_space_dimension')

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
dset = datasets.IIPDataset('/home/pikey/Data/II/crop_part1','train',crop_ratio=(0,0))
dldr = DataLoader(dset,batch_size=args.bs,shuffle=True,num_workers=16,drop_last=True)
print('data :{}'.format('crop_part1,train'))

#todo --------------------------------modelpreparation------------------------------------
torch.manual_seed(args.seed)
m = ae.AE(args.z_dim).to(gpu)
opt = optim.Adam(m.parameters(),lr=args.lr)
print('model done')

#todo --------------------------------logger init-------------------------------------
if os.path.exists(os.path.join(args.log_dir, 'train_result', 'img')) == False:
	os.makedirs(os.path.join(args.log_dir, 'train_result', 'img'))
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train_result'))
print('logger dir:{}'.format(os.path.join(args.log_dir, 'train_result', 'img')))

#todo"""--------------------------------train------------------------------------"""
def train():
	print('train begin')

	# train
	step = 0
	for epoch in trange(args.epoch):
		bt = 0
		for true, flaw, region in dldr:
			bt += 1
			step += 1
			#todo"""--------------------------------forward------------------------------------"""
			true = true.to(gpu)
			rec = m(true)

			#todo"""--------------------------------train------------------------------------"""
			loss = ae.compute_loss(rec,true)
			opt.zero_grad()
			loss.backward()
			opt.step()

			#todo"""--------------------------------savedata------------------------------------"""
			# log
			print('loss_rec:{}'.format(loss))
			if step % 50 == 0:\

				img = torchvision.utils.make_grid(
					[true[0].cpu(),
					 m.rec[0].view(3, datasets.IIPDataset.height, datasets.IIPDataset.width).cpu()],
					nrow=2)
				writer.add_image('results', img, step)
				torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img',
															   'epoch{}_batch{}.png'.format(epoch, bt)))

		if not os.path.exists(os.path.join(args.log_dir, 'model')):
			os.makedirs(os.path.join(args.log_dir, 'model'))
		torch.save(m.state_dict(), os.path.join(args.log_dir, 'model', 'epoch{}.pkl'.format(epoch)))


if __name__ == '__main__':
	train()