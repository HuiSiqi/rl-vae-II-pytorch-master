import torch
from torch import nn

def  conv_layer(inchannel,config = '1K5D1S1C64B1'):
	#the config be like 1K5D1S1C64B1
	#denote conv or deconv,kernel size,dilation,stride,channel,need batch_norm
	assert type(config)==type('str')
	conv,config = config.split('K')
	conv = bool(conv)
	kernel_size,config = config.split('D')
	kernel_size = int(kernel_size)
	dilation,config = config.split('S')
	dilation = int(dilation)
	stride,config = config.split('C')
	stride = int(stride)
	if stride==0:
		stride = 1
		padding = 0
	else:
		padding = 1
	channel,batch_norm = config.split('B')
	channel = int(channel)
	batch_norm = bool(batch_norm)

	layer = []
	if conv:
		layer.append(nn.Conv2d(inchannel,channel,kernel_size
		                       ,stride,padding,dilation,bias=not batch_norm))
	else:
		layer.append(nn.ConvTranspose2d(inchannel,channel,kernel_size,stride,padding,bias=not batch_norm))
	layer.append(nn.ReLU())
	layer.append(nn.BatchNorm2d(channel))

	return nn.Sequential(*layer)

