#todo"""-----------------import normal package-------------------------"""
import sys
sys.path.append('/home/pikey/PycharmProjects/rl-vae-II-pytorch-master')
import torch,torchvision,os,json,argparse,utils,setproctitle, tqdm
from tensorboardX import SummaryWriter
from tqdm import trange

#todo------------------------------import user package-------------------------------
from environment import envs
from DDPG import agent
import pylib
from train.ddpg.conv.one_step import E30config
#todo get path
dirname,filename = os.path.split(os.path.abspath(__file__))

#todo -------------------------------init----------------------------------------
#todo args
args = pylib.args()
args.log_dir+=filename.split('.')[0]
pylib.args_to_yaml(args.log_dir,args)
#todo writter
writer = SummaryWriter()
#todo environment
e = envs.OneStep(args)
#todo agent
player = agent.Agent(args,config='conv')

step = 0
# todo -----------------------------warm up--------------------------------------
def wmup():
	for i in trange(args.wmup,desc='WARM UP'):
		last_state = e.reset()
		#todo agent
		with torch.no_grad():
			action = player.action(last_state)
		e.check_action(action)
		action +=args.sigma*torch.ones_like(action)
		state,reward,done = e.step(action)
		player.store(last_state.detach().cpu().numpy(),
		             action.detach().cpu().numpy(),
		             reward,
		             state.detach().cpu().numpy(),
		             done)

def search(batch):
	total_r = 0
	e.show_flag = True
	for i in range(batch):
		last_state = e.reset()
		#todo agent
		with torch.no_grad():
			action = player.action(last_state)
		action +=args.sigma*torch.ones_like(action)
		state,reward,done = e.step(action)

		player.store(last_state.detach().cpu().numpy(),
		             action.detach().cpu().numpy(),
		             reward,
		             state.detach().cpu().numpy(),
		             done)
		if i==0:e.show_flag = False
	total_r+=reward.mean()
	print('mean_reward:{}'.format(total_r/batch))

def test():
	last_state = e.reset()
	# todo agent

	with torch.no_grad():
		action = player.action(last_state)
	state, reward, done = e.step(action)
	buffer = e.log()
	true,true_rec, noisy, rec, mix,fake = buffer
	img = torchvision.utils.make_grid(
		[true[0].cpu(),true_rec[0].cpu(), noisy[0].cpu(), rec[0].cpu(),mix[0].cpu(), fake[0].cpu()], nrow=6)
	img = utils.rescale(img)  # change image to 0-1
	# writer.add_image('results', img, step)
	torchvision.utils.save_image(img, os.path.join(args.log_dir, 'train_result', 'img',
	                                               'step:{}.png'.format(step)))
	writer.add_scalar('reward',reward.mean(),step)

def save_model():
	torch.save(player.actor_target.state_dict(),os.path.join(args.log_dir,'model','actor','step:{}'.format(step)))
	torch.save(player.actor_target.state_dict(),os.path.join(args.log_dir,'model','critic','step:{}'.format(step)))
def train():
	global step
	for i in trange(int(args.epoch),desc='Loop'):
		search(5)
		for j in range(10):
			player.learn()
			step+=1

		if i%10==0:
			test()
		if i%int(args.epoch/10):
			save_model()

if __name__ == '__main__':
	utils.mkdir(os.path.join(args.log_dir, 'train_result', 'img'))
	utils.mkdir(os.path.join(args.log_dir, 'model', 'actor'))
	utils.mkdir(os.path.join(args.log_dir, 'model', 'critic'))
	wmup()
	train()