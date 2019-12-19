#todo"""-----------------import normal package-------------------------"""
import sys
sys.path.append('/home/pikey/PycharmProjects/rl-vae-II-pytorch-master')
import torch,torchvision,os,json,argparse,utils,setproctitle, tqdm
from tensorboardX import SummaryWriter

#todo"""--------------------------------import user package------------------------------------"""
from DDPG import agent
from environment import one_step_v2

#todo get path
dirname,filename = os.path.split(os.path.abspath(__file__))

'''
modify gpu,img_num,face_per_img
if u want to change the model please modify dst_cp, model_dir,max_a,step
'''
#todo"""--------------------------------args------------------------------------"""
parser = argparse.ArgumentParser(description='train ae')

log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')
env = parser.add_argument_group('environment')
memo = parser.add_argument_group('memo')

parser.add_argument('--gpu',type=int,default=-1,help='gpu id')
parser.add_argument('--seed',type=int,default=10000,help='random seed')
parser.add_argument('--img_num',type=int,default=100,help='img number to be saved')
parser.add_argument('--face_per_img',type=int,default=16,help='reconstruction face per image')

log.add_argument('--log-dir',type=str,default=os.path.join('/home/pikey/Data/II/test',filename),help='log directory')

train.add_argument('--bs',type=int,default=256,metavar='BS',help='training batch size')
train.add_argument('--lr',type=float,default=3e-4,metavar='lr',help='learning rate')

env.add_argument('--ae',type=str,default='ae',help='the autoencoder model')
env.add_argument('--wmse',type=float,default=1,help='the weight of reconstruction reward')
env.add_argument('--wpsnr',type=float,default=1,help='the weight of reconstruction reward')
env.add_argument('--wssim',type=float,default=1,help='the weight of reconstruction reward')
env.add_argument('--wl2',type=float,default=1,help='the weight of feature vector reward')
env.add_argument('--wconsis',type=float,default=1,help='the weight of consistance reward ')
env.add_argument('--dim_a',type=int,default=64,help='action dim')
env.add_argument('--dim_s',type=int,default=64,help='state dim')
env.add_argument('--model-root',type=str,default='/home/pikey/Data/II/pretrained_model',help='state dim')
env.add_argument('--dst',type=str,default='/home/pikey/Data/II/crop_part1',help="dataset root")
env.add_argument('--dst-random-noise',type=bool,default=False,help="use random noise")
env.add_argument('--dst-idx',type=str,default='test',help="choose from ['train','test','all']")
env.add_argument('--dst_cp',type=list,default=[0.4,0.4],help="the missing scale of the height and width")
env.add_argument('--dst-bs',type=int,default=256,help="dataset sample batchsize")

model.add_argument('--h1_dim',type=int,default=512,help="h1 dimension")
model.add_argument('--h2_dim',type=int,default=256,help="h2 dimension")
model.add_argument('--dim_f',type=int,default=256,help='state dim')
model.add_argument('--sigma',type=float,default=0.01,help="action noise variance")
model.add_argument('--gamma',type=float,default=0.01,help="reward decay rate")
model.add_argument('--max_a',type=float,default=30,help="max_action")
model.add_argument('--tau',type=float,default=0.05,help="soft update weight")
model.add_argument('--model-dir',type=str,default=os.path.join('/home/pikey/Data/II/ddpg/fc/one_step','ae_a64.py','8','model'),help="actor critic model directory")
model.add_argument('--step',type=str,default='35000',help="check point step")

args = parser.parse_args()

torch.manual_seed(args.seed)
#todo save args
config = vars(args)
#create roots
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

#todo"""--------------------------------envpreparation------------------------------------"""
e = one_step_v2.env(args)
print('--------------------------------------------')
print('environment done')
print('crop:{}'.format(args.dst_cp))
print('ae:{}       bs:{}       wrec,wl2,wconsis:{},{},{}'.format(args.ae,args.dst_bs,args.wmse,args.wl2,args.wconsis))
print('--------------------------------------------')

#todo --------------------------------modelpreparation------------------------------------
player = agent.Agent(args.dim_s,args.dim_a,gpu,args)
actor_state_dict = torch.load(os.path.join(args.model_dir,'actor','step'+args.step+'.pkl'))
player.actor_target.load_state_dict(actor_state_dict)
player.actor_target.eval()

#todo ----------------------------------set title-----------------------------------
setproctitle.setproctitle('ddpg test gpu:{}'.format(args.gpu))

#todo ----------------------------------exploration-----------------------------------
# def explore():
	# for i in range():
	# s = e.observe()
	# a = player.actor_target(torch.Tensor(s).to(gpu))
	# a_ = a+torch.rand_like(a)*args.sigma
	# s, a_, r, s_, t = e.step(a_)
	# # player.store(s, a_, r, s_, t)
	# print('reward:{}    action_norm:{}'.format(r.mean(),a.norm(2,dim=1,keepdim=True)[0].detach().cpu().numpy()))
	# print('action:{}'.format(a[0][:4].detach().cpu().numpy()))
	# print('a_:{}'.format(a_[0][:2]))

#todo ----------------------------------log-------------------------------------
def log(imgname):

	s = e.observe()
	a = player.actor_target(torch.Tensor(s).to(gpu))
	# a += torch.rand_like(a) * args.sigma
	s, a, r, s_, t = e.step(a)
	true,noisy,rec,fake,mix = e.buffer
	img = torch.cat((true,noisy,rec,fake,mix),dim=1)[0:args.face_per_img]
	img = torchvision.utils.make_grid(img,nrow=args.face_per_img)
	torchvision.utils.save_image(img, os.path.join(args.log_dir,
												   '{}.jpg'.format(imgname)))


if __name__ == '__main__':
	for i in range(args.img_num):
		log(i)
	# os.makedirs(os.path.join(args.log_dir, 'test'),exist_ok=True)
	#
	# explore(1000)

