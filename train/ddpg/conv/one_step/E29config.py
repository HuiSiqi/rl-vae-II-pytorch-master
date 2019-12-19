import pylib

pylib.arg('--gpu',type=int,default=2)
pylib.arg('--seed',type=int,default=2)
pylib.arg('--log-dir',type=str,default='/home/pikey/Data/II/ddpg/',help='log directory')

#todo ---------------------------env config--------------------------------
pylib.arg('--dst',default='celeba',choices=['dataset','celeba'])
pylib.arg('--dst-cp',type=tuple,default=(0.4,0.4))
pylib.arg('--dst-bs',type=int,default=8,help='dataset batch size')

pylib.arg('--feature-state',type=bool,default=False,help='dataset batch size')

#should be same as ae train config
pylib.arg('--ae',type=str,default='ae',help='encoder decoder')
pylib.arg('--ae-conv',type=int,default=5,help='conv number')
pylib.arg('--ae-dilation',type=int,default=4,help='dilation conv number')
pylib.arg('--ae-model',type=str,default='/home/pikey/Data/II/pretrained_model/'+'ae.pkl')
pylib.arg('--ae-neck-size',type=tuple,default=(16,16,8),help='(channel,h,w)')
pylib.arg('--img-size',type=int,default=256,help='dataset batch size')

#todo ----------------------agent config------------------------------------
pylib.arg('--lr',type=float,default=1e-4,help='decay of reward weight')
pylib.arg('--gamma',type=float,default=0.9,help='decay of reward weight')
pylib.arg('--tau',type=float,default=0.05,help='model soft update rate')
pylib.arg('--sigma',type=float,default=0.1,help='init noise value')
pylib.arg('--bs',type=int,default=10,help='batch size')
#todo actor config
pylib.arg('--state-size',type=tuple,default=(16,16,8),help='(channel,h,w)')

pylib.arg('--action-size',type=tuple,default=(16,16,8),help='(channel,h,w)')
pylib.arg('--n-conv',type=int,default=4,help='only affects the depth of net, irrelevent to the shape')
pylib.arg('--n-dilation',type=int,default=3,help='only affects the depth of net, irrelevent to the shape')
pylib.arg('--max-a',type=int,default=2,help='max action value')

#todo critic config
pylib.arg('--dim-f',type=int,default=1024,help='feature to create value')

#todo -----------------------