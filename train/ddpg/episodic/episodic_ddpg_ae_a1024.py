#todo"""-----------------import normal package-------------------------"""
import sys
sys.path.append('/home/pikey/PycharmProjects/rl-vae-II-pytorch-master')
import torch,torchvision,os,json,argparse,utils,setproctitle, tqdm
from tensorboardX import SummaryWriter

#todo"""--------------------------------import user package------------------------------------"""
from DDPG import agent
from environment import env_vae

#todo get path
dirname,filename = os.path.split(os.path.abspath(__file__))

#todo"""--------------------------------args------------------------------------"""
parser = argparse.ArgumentParser(description='train ae')

log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')
env = parser.add_argument_group('environment')
memo = parser.add_argument_group('memo')

parser.add_argument('--gpu',type=int,default=0,help='gpu id')
parser.add_argument('--seed',type=int,default=10000,help='random seed')

train.add_argument('--epoch',type=int,default=400,metavar='TS',help='training steps')
train.add_argument('--bs',type=int,default=256,metavar='BS',help='training batch size')
train.add_argument('--lr',type=float,default=1e-3,metavar='lr',help='learning rate')

log.add_argument('--log-dir',type=str,default=os.path.join('/home/pikey/Data/II','ddpg_ae_a1024_max_a40'),help='log directory')

env.add_argument('--ae',type=str,default='ae',help='the autoencoder model')
env.add_argument('--wrec',type=float,default=2,help='the weight of reconstruction reward')
env.add_argument('--wl2',type=float,default=1,help='the weight of feature vector reward')
env.add_argument('--wconsis',type=float,default=1,help='the weight of consistance reward ')
env.add_argument('--dim_a',type=int,default=1024,help='action dim')
env.add_argument('--dim_s',type=int,default=1024,help='state dim')
env.add_argument('--model-root',type=str,default='/home/pikey/Data/II/pretrained_model',help='state dim')
env.add_argument('--dst',type=str,default='/home/pikey/Data/II/crop_part1',help="dataset root")
env.add_argument('--dst-idx',type=str,default='train',help="choose from ['train','test','all']")
env.add_argument('--dst_cp',type=list,default=[0.4,0.4],help="the missing scale of the height and width")
env.add_argument('--dst-bs',type=int,default=256,help="dataset sample batchsize")
env.add_argument('--epslen',type=int,default=5,help="dataset sample batchsize")

model.add_argument('--h1_dim',type=int,default=512,help="h1 dimension")
model.add_argument('--h2_dim',type=int,default=512,help="h2 dimension")
model.add_argument('--sigma',type=float,default=0.01,help="action noise variance")
model.add_argument('--gamma',type=float,default=0.01,help="reward decay rate")
model.add_argument('--max_a',type=float,default=1,help="max_action")
model.add_argument('--tau',type=float,default=0.05,help="soft update weight")

memo.add_argument('--limit',type=int,default=5e4,help="memory size")
memo.add_argument('--wmup',type=int,default=3,help="data number before to train ")


args = parser.parse_args()
torch.manual_seed(args.seed)
#todo save args
config = vars(args)
#create roots
if not os.path.exists(args.log_dir):
	os.makedirs(args.log_dir)
if not os.path.exists(args.model_root):
	os.makedirs(args.model_root)
with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
	json.dump(config, f, cls=utils.DataEnc, indent=2)

#todo"""--------------------------------gpu preparation------------------------------------"""
if args.gpu == -1:
	gpu = 'cpu'
else:
	gpu = args.gpu
print('gpu :{}'.format(gpu))

#todo"""--------------------------------envpreparation------------------------------------"""
e = env_vae.env(args)
print('--------------------------------------------')
print('environment done')
print('crop:{}'.format(args.dst_cp))
print('ae:{}       bs:{}       wrec,wl2,wconsis:{},{},{}'.format(args.ae,args.dst_bs,args.wrec,args.wl2,args.wconsis))
print('--------------------------------------------')
#todo -----------------------------------memorypreparation--------------------------------
print('memory limit:{} warmup:{}'.format(args.limit,args.wmup))
#todo --------------------------------modelpreparation------------------------------------
player = agent.Agent(2*args.dim_s,args.dim_a,gpu,args)

#todo --------------------------------logger init-------------------------------------
if os.path.exists(os.path.join(args.log_dir, 'train_result','img')) == False:
	os.makedirs(os.path.join(args.log_dir, 'train_result', 'img'))
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train_result'))
print('logger dir:{}'.format(os.path.join(args.log_dir, 'train_result')))

#todo ----------------------------------set title-----------------------------------
setproctitle.setproctitle('ddpg gpu:{}'.format(args.gpu))

step=0
noise = utils.decay_scale(args.sigma,1e-4*args.sigma,1e4)
print('noise done')
#todo ----------------------------------warm up-----------------------------------
def warm_up():
	print('---------------------------warm up-----------------------------')
	explore(args.wmup)

#todo ----------------------------------exploration-----------------------------------
def explore(n):
	for i in range(n):
		e.reset()
		for j in range(args.epslen):
			s = e.observe()
			a = player.actor_target(torch.Tensor(s).to(gpu))
			a_ = a+torch.rand_like(a)*args.sigma
			s, a_, r, s_, t = e.step(a_)
			player.store(s, a_, r, s_, t)
			print('reward:{}    action_norm:{}'.format(r.mean(),a.norm(2,dim=1,keepdim=True)[0].detach().cpu().numpy()))
			print('action:{}'.format(a[0][:4].detach().cpu().numpy()))
			print('a_:{}'.format(a_[0][:4]))

#todo ----------------------------------train-----------------------------------
#todo ----------------------------------log-------------------------------------
def log(img=False,model=False):
	if img:
		s = e.observe()
		a = player.actor_target(torch.Tensor(s).to(gpu))
		# a += torch.rand_like(a) * args.sigma
		s, a, r, s_, t = e.step(a)
		true,noisy,rec,fake = e.buffer
		img = torchvision.utils.make_grid(
			[true[0].cpu(),noisy[0].cpu(),rec[0].cpu(),fake[0].cpu()],nrow=4)
		writer.add_image('results', img, step)
		torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result','img',
													   'step:{}.png'.format(step)))

	if model:
		if not os.path.exists(os.path.join(args.log_dir, 'model','actor')):
			os.makedirs(os.path.join(args.log_dir, 'model','actor'))
		if not os.path.exists(os.path.join(args.log_dir, 'model','critic')):
			os.makedirs(os.path.join(args.log_dir, 'model','critic'))
		torch.save(player.actor_local.state_dict(), os.path.join(args.log_dir, 'model','actor', 'step{}.pkl'.format(step)))
		torch.save(player.critic_local.state_dict(), os.path.join(args.log_dir, 'model','critic', 'step{}.pkl'.format(step)))

if __name__ == '__main__':
	warm_up()
	for i in tqdm.trange(100000):
		explore(1)
		if i%50==0:
			log(img=True)
		if i%500==0:
			log(model=True)
		for j in range(20):
			player.learn()
		step +=1
	# os.makedirs(os.path.join(args.log_dir, 'test'),exist_ok=True)
	#
	# explore(1000)

