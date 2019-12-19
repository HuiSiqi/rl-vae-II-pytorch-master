#todo"""-----------------import normal package-------------------------"""
import torch,torchvision,os,json,argparse,utils,setproctitle
from torch.utils.data import DataLoader
from tqdm import trange
from torch import optim
from tensorboardX import SummaryWriter

#todo"""--------------------------------import user package------------------------------------"""
from vae_wgan import vae_wgan
from datasets import datasets
from lwgan import lwgan
#todo"""--------------------------------args------------------------------------"""
parser = argparse.ArgumentParser(description='train ae')

log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')

parser.add_argument('--gpu',type=int,default=0,help='gpu id')
parser.add_argument('--seed',type=int,default=0,help='random seed')

train.add_argument('--epoch',type=int,default=400,metavar='TS',help='training steps')
train.add_argument('--bs',type=int,default=256,metavar='BS',help='training batch size')
train.add_argument('--lr',type=float,default=1e-3,metavar='lr',help='learning rate')

log.add_argument('--log-dir',type=str,default='/home/pikey/Data/II/lwgan_vae_wgan',help='log directory')

model.add_argument('--z-dim',type=int,default=1024,metavar='dz',help='latent_space_dimension')
model.add_argument('--g-dim',type=int,default=8,metavar='dz',help='latent_space_dimension')
model.add_argument('--clip-w',type=float,default=0.1,metavar='c',help='the abs of max weights')

args = parser.parse_args()

#todo save args
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
AE = vae_wgan.VAE(args.z_dim)
AE.load_state_dict(torch.load('/home/pikey/Data/II/vae_wgan/model/vae/epoch399.pkl'))
AE.to(gpu)
AE.eval()
m = lwgan.LWGAN(args.g_dim,args.z_dim).to(gpu)
gen_opt = optim.Adam(m.generator.parameters(),lr=args.lr)
dis_opt = optim.Adam(m.discriminator.parameters(),lr=args.lr)
print('model done')

#todo --------------------------------logger init-------------------------------------
if os.path.exists(os.path.join(args.log_dir, 'train_result','img')) == False:
	os.makedirs(os.path.join(args.log_dir, 'train_result', 'img'))
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train_result'))
print('logger dir:{}'.format(os.path.join(args.log_dir, 'train_result', 'img')))

#todo ----------------------------------set title-----------------------------------
setproctitle.setproctitle('lwgan_vae_gan gpu:{}'.format(args.gpu))

#todo"""--------------------------------train------------------------------------"""
def train():
	print('train begin')
	noise= utils.decay_scale(1e-2,1e-8,10000)
	# train
	step = 0
	for epoch in trange(args.epoch):
		bt = 0
		for true, flaw, region in dldr:
			bt += 1
			step += 1
			#todo"""--------------------------------forward------------------------------------"""
			true = true.to(gpu)

			seed =  torch.randn(args.bs,args.g_dim).to(gpu)

			#todo get the fake judge
			fake_f = m.gen(seed)
			fake_img = AE.decode(fake_f)
			# fake_f = AE.enc(fake_img)
			fake_f = fake_f+torch.rand_like(fake_f)*noise()
			noise.step()
			judge_fake = m.dis(fake_f)

			rec = AE(true)

			judge_real = m.dis(AE.z)

			#todo"""--------------------------------train------------------------------------"""
			gen_loss = lwgan.generator_loss(judge_fake)
			dis_loss = lwgan.discriminator_loss(judge_real,judge_fake)

			gen_opt.zero_grad()
			gen_loss.backward(retain_graph=True)
			gen_opt.step()

			dis_opt.zero_grad()
			dis_loss.backward()
			dis_opt.step()

			#clip the weights
			lwgan.clip_weight(m.discriminator,args.clip_w)

			#todo"""--------------------------------savedata------------------------------------"""
			# log
			print('loss_gen:{},     loss_dis:{}'.format(gen_loss,dis_loss))
			writer.add_scalar('loss_gen',gen_loss,step)
			writer.add_scalar('loss_dis',dis_loss,step)
			if step % 50 == 0:

				img = torchvision.utils.make_grid(
					[
					 fake_img[0].cpu()],
					nrow=1)
				writer.add_image('results', img, step)
				torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img',
															   'epoch{}_batch{}.png'.format(epoch, bt)))

		if not os.path.exists(os.path.join(args.log_dir, 'model')):
			os.makedirs(os.path.join(args.log_dir, 'model'))
		torch.save(m.state_dict(), os.path.join(args.log_dir, 'model', 'epoch{}.pkl'.format(epoch)))

if __name__ == '__main__':
	train()