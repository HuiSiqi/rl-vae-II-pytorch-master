#todo"""-----------------import normal package-------------------------"""
import sys
sys.path.append('/home/pikey/PycharmProjects/rl-vae-II-pytorch-master')
import torch,torchvision,os,json,argparse,utils,setproctitle, tqdm
from tensorboardX import SummaryWriter

#todo------------------------------import user package-------------------------------
from environment import base
from DDPG import agent
import pylib
from train.ddpg.conv.one_step import E27config
#todo get path
dirname,filename = os.path.split(os.path.abspath(__file__))

args = pylib.args()
args.log_dir+=filename
pylib.args_to_yaml(args.log_dir,args)
e = base.E1(args)
state = e.reset()
player = agent.Agent(args,config='conv')

action = player.action(state)

