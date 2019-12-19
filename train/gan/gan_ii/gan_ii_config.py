import pylib

pylib.arg('--gpu',type=int,default=0)
pylib.arg('--multi-gpu',type=tuple,default=(0,1,2,3))
pylib.arg('--seed',type=int,default=2000)
pylib.arg('--log-dir',type=str,default='/home/pikey/Data/II/gan/',help='log directory')

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

#todo -----------------------train ----------------------------------------
pylib.arg('--epoch',type=int,default=1e5,help='train steps')
pylib.arg('--lr',type=float,default=1e-3,help='learning rate')
pylib.arg('--gamma',type=float,default=10,help='gp weight')
pylib.arg('--beta',type=float,default=0.5,help='adam parameters')
