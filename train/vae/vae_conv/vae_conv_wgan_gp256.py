#todo"""-----------------import normal package-------------------------"""
import sys
sys.path.append('/home/pikey/PycharmProjects/rl-vae-II-pytorch-master')
import torch,torchvision,os,json,argparse,setproctitle,utils
from torch.utils.data import DataLoader
from tqdm import trange
from torch import optim
from tensorboardX import SummaryWriter
#todo"""--------------------------------import user package------------------------------------"""
from vae_wgan_gp import vae_wgan_gp
from datasets import datasets

#todo get path
dirname,filename = os.path.split(os.path.abspath(__file__))

#todo"""--------------------------------args------------------------------------"""
parser = argparse.ArgumentParser(description='train ae')

log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')

parser.add_argument('--gpu',type=int,default=2,help='gpu id')
parser.add_argument('--seed',type=int,default=0,help='random seed')

train.add_argument('--epoch',type=int,default=400,metavar='TS',help='training steps')
train.add_argument('--bs',type=int,default=64,metavar='BS',help='training batch size')
train.add_argument('--lr',type=float,default=3*1e-4,metavar='lr',help='learning rate')

log.add_argument('--log-dir',type=str,default=os.path.join('/home/pikey/Data/II',filename),help='log directory')

model.add_argument('--z-dim',type=int,default=128,metavar='dz',help='latent_space_dimension')
model.add_argument('--gamma',type=float,default=1,metavar='gamma',help='train weight of decoder')
model.add_argument('--beta',type=float,default=1,metavar='beta',help='weight of discriminator learning rate')

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
vae = vae_wgan_gp.VAE(args.z_dim,config='vae_conv')
vae.to(gpu)
dis = vae_wgan_gp.Discriminator().to(gpu)

dec_opt = optim.Adam(vae.decoder.parameters(),lr=args.lr)
enc_opt = optim.Adam(vae.encoder.parameters(),lr=args.lr)
dis_opt = optim.Adam(dis.parameters(),lr=args.lr)
print('model done')

#todo --------------------------------logger init-------------------------------------
if os.path.exists(os.path.join(args.log_dir, 'train_result','img')) == False:
	os.makedirs(os.path.join(args.log_dir, 'train_result', 'img'))
	os.makedirs(os.path.join(args.log_dir, 'train_result', 'img','rec'))
	os.makedirs(os.path.join(args.log_dir, 'train_result', 'img','gen'))

writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train_result'))
print('logger dir:{}'.format(os.path.join(args.log_dir, 'train_result')))
#todo ----------------------------------set title-----------------------------------
setproctitle.setproctitle('vae_conv_wgan_gp gpu:{}'.format(args.gpu))

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
			x = true.to(gpu)

			tild_x = vae(x)
			hat_x = vae.decode(vae.prior.sample(vae.z_mean.size()).to(gpu))
			judge_hat_x = dis(hat_x)
			judge_tild_x = dis(tild_x)
			judge_fake = torch.cat([judge_hat_x, judge_tild_x], dim=0)
			judge_x = dis(x)
			#todo ---------------------------------train-------------------------------------
			enc_opt.zero_grad()


			loss_rec,loss_kld = vae_wgan_gp.vae_loss(tild_x,x,vae.z_mean,vae.z_logsigma,mse=True)
			loss_dis = vae_wgan_gp.discriminator_loss(judge_x,judge_fake)

			#train encoder
			enc_opt.zero_grad()
			(loss_rec+loss_kld).backward(retain_graph=True)
			enc_opt.step()

			#train decoder
			dec_opt.zero_grad()
			loss_dec = args.gamma*loss_rec-loss_dis
			loss_dec.backward(retain_graph=True)
			dec_opt.step()

			# train discriminator
			dis_opt.zero_grad()
			gp = vae_wgan_gp.gradient_penalty(dis,x,hat_x)
			loss_dis = vae_wgan_gp.discriminator_loss(judge_x, judge_fake)
			loss_dis+=gp*args.gamma
			loss_dis.backward()
			dis_opt.step()

			#todo"""--------------------------------savedata------------------------------------"""
			# log
			print('loss_rec:{},     loss_dec:{},       loss_dis:{}'.format(loss_rec,loss_dec,loss_dis))

			writer.add_scalar('loss_rec',loss_rec,step)
			writer.add_scalar('loss_dec',loss_dec,step)
			writer.add_scalar('loss_dis',loss_dis,step)
			if step % 50 == 0:\
				#todo save reconstructed image
				img = torchvision.utils.make_grid(
					[true[0].cpu(),
					 tild_x[0].cpu()],
					nrow=2)
				writer.add_image('results', img, step)
				torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img','rec',
															   'epoch{}_batch{}.png'.format(epoch, bt)))

				# todo save generated image
				img = torchvision.utils.make_grid(
					[
						hat_x[0].cpu()],
					nrow=1)
				writer.add_image('results', img, step)
				torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img', 'gen',
				                                               'epoch{}_batch{}.png'.format(epoch, bt)))

		#todo save model
		utils.save_model(vae,os.path.join(args.log_dir, 'model','vae'),'epoch{}.pkl'.format(epoch))
		utils.save_model(dis,os.path.join(args.log_dir, 'model','discriminator'),'epoch{}.pkl'.format(epoch))

if __name__ == '__main__':
	train()