import pylib

pylib.arg('--gpu',type=int,default=1)
pylib.arg('--multi-gpu',type=tuple,default=(0,1,2,3))
pylib.arg('--seed',type=int,default=2)
pylib.arg('--log-dir',type=str,default='/home/pikey/Data/II/ddpg/',help='log directory')

#todo ---------------------------env config--------------------------------
pylib.arg('--dst',default='celeba',choices=['dataset','celeba'])
pylib.arg('--dst-cp',type=tuple,default=(0.4,0.4))
pylib.arg('--dst-bs',type=int,default=32,help='dataset batch size')
pylib.arg('--dst-nwks',type=int,default=16,help='dataset number workers')

pylib.arg('--feature-state',type=bool,default=False,help='dataset batch size')

#should be same as ae train config
pylib.arg('--ae',type=str,default='ae',help='encoder decoder')
pylib.arg('--ae-conv',type=int,default=4,help='conv number')
pylib.arg('--ae-dilation',type=int,default=3,help='dilation conv number')
pylib.arg('--ae-model',type=str,default='/home/pikey/Data/II/pretrained_model/'+'ae.pkl')
pylib.arg('--ae-neck-size',type=tuple,default=(8,16,16),help='(channel,h,w)')
pylib.arg('--img-size',type=int,default=256,help='dataset batch size')
#todo reward weight
pylib.arg('--w_l1',type=float,default=10,help='decay of reward weight')
pylib.arg('--w_dis',type=float,default=1,help='decay of reward weight')
pylib.arg('--w_local_dis',type=float,default=1,help='decay of reward weight')
pylib.arg('--w_psnr',type=float,default=10,help='decay of reward weight')
pylib.arg('--w_ssim',type=float,default=1,help='decay of reward weight')
pylib.arg('--w_local_ssim',type=float,default=1,help='decay of reward weight')

#todo ----------------------agent config------------------------------------
pylib.arg('--limit',type=int,default=5e3,help='replay buffer length')
#todo actor config
pylib.arg('--state-size',type=tuple,default=(8,16,16),help='(channel,h,w)')

pylib.arg('--action-size',type=tuple,default=(8,16,16),help='(channel,h,w)')
pylib.arg('--n-state-conv',type=int,default=4,help='only affects the depth of actor and critic image feature net, irrelevent to the shape, '
                                             'should be decided by the state size')
pylib.arg('--n-feature-conv',type=int,default=4,help='only affects the depth of critic cmp net, irrelevent to the shape, '
                                             'should be decided by the feature size')
pylib.arg('--n-dilation',type=int,default=3,help='only affects the depth of cmp net of actor, irrelevent to the shape')
pylib.arg('--max-a',type=int,default=10,help='max action value')

#todo critic config
pylib.arg('--dim-f',type=int,default=1024,help='feature to create value')

#todo -----------------------train ----------------------------------------
pylib.arg('--epoch',type=int,default=1e5,help='train steps')
pylib.arg('--lr',type=float,default=1e-3,help='learning rate')
pylib.arg('--gamma',type=float,default=0.9,help='decay of reward weight')
pylib.arg('--tau',type=float,default=0.05,help='model soft update rate')
pylib.arg('--sigma',type=float,default=0.5,help='init noise value')
pylib.arg('--bs',type=int,default=48,help='batch size')
pylib.arg('--wmup',type=int,default=3,help='warm up batches')
