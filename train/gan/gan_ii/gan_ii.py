#todo --------------------general packadge-----------------------
import sys,os,torch,numpy as np,pylib,tqdm,torchvision,utils
from torch import nn,optim
from torch.utils.data import DataLoader
sys.path.append('/home/pikey/PycharmProjects/rl-vae-II-pytorch-master')
#todo --------------------usr packadge---------------------------
from train.gan import gan_ii_config
from wgan_gp import wgan_gp
from ae import ae
from datasets import celeba
#todo get args
args = pylib.args()
#todo get path
dirname,filename = os.path.split(os.path.abspath(__file__))

args.log_dir+=filename.split('.')[0]
pylib.args_to_yaml(args.log_dir,args)

#todo -------------------------init-----------------------------
#todo model
model = wgan_gp.WGAN_GP(1,3,6)
gen = ae.AE(3,256,3,2,4,2,False)
model.generator = gen
model.to(args.gpu)
if len(args.multi_gpu)>1:
	model = nn.DataParallel(model,device_ids=args.multi_gpu)

	gen_opt = optim.Adam(model.module.generator.parameters(),lr=args.lr,betas=(args.beta,0.999))
	dis_opt = optim.Adam(model.module.discriminator.parameters(),lr=args.lr,betas=(args.beta,0.999))
else:
	gen_opt = optim.Adam(model.generator.parameters(),lr=args.lr,betas=(args.beta,0.999))
	dis_opt = optim.Adam(model.discriminator.parameters(),lr=args.lr,betas=(args.beta,0.999))
#todo dataset
dst = celeba.IIPDataset('/home/pikey/DataSet/celeba','train',256,(0.4,0.4))
dldr = DataLoader(dst,batch_size=args.dst_bs,shuffle=True,num_workers = args.dst_nwks,drop_last=True)

#todo ----------------------train --------------------------------

def train(epoch):
	for i in tqdm.trange(epoch):
		bt=0
		for true,noise,bbox in tqdm.trange(dldr,desc='Epoch'):
			gen_loss,dis_loss,fake = model((noise,true))
			gen_loss,dis_loss = gen_loss.mean(),dis_loss.mean()
			gen_opt.zero_grad()
			gen_loss.backward()
			gen_opt.step()

			dis_opt.zero_grad()
			dis_loss.backward()
			dis_opt.step()

			bt+=1
			print('gen_loss:{}  dis_loss:{}'.format(gen_loss.item(),dis_loss.item()))
			if bt%2000==0:
				img = torchvision.utils.make_grid(
					[true[0].cpu(),
					 fake[0].detach().cpu()],
					nrow=2)
				img = utils.rescale(img)
				# writer.add_image('results', img, step)
				torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img',
				                                               'epoch{}_batch{}.png'.format(i, bt)))


if __name__ == '__main__':
	train(20)

