#todo -----------------------load model packages--------------------------
from vae_wgan import vae_wgan
from vae_wgan_gp import vae_wgan_gp
from vae_gan import vae_gan
from ae import ae as AE
from lgan import lgan
from vae_wgan_gp import vae_wgan_gp
from lwgan_gp import lwgan_gp
# todo load models
def load_model(args):
	if args.ae == 'vae_gan':
		ae = vae_gan.VAE(args.dim_s, 'conv')
		dis = vae_wgan_gp.Discriminator()
	elif args.ae == 'vae_wgan':
		ae = vae_wgan.VAE(args.dim_s, 'conv')
		dis = vae_wgan_gp.Discriminator()
	elif args.ae == 'vae_wgan_gp':
		ae = vae_wgan_gp.VAE(args.dim_s, config='conv')
		dis = vae_wgan_gp.Discriminator()
	elif args.ae == 'ae':
		ae = AE.AE(args.dim_s, 'conv')
		dis = vae_wgan_gp.Discriminator()
	elif args.ae == 'vae':
		ae = vae_wgan_gp.VAE(args.dim_s, 'conv')
		dis = vae_gan.Discriminator()
	else:
		raise ValueError("model:{} should be one of vae_gan,vae_wgan,ae".format(args.ae))
	if args.lgan:
		lgan = lwgan_gp.LWGAN_GP_Conv(args.dim_g,args.dim_s,args.n_conv)
	else:
		lgan = None
	return ae,lgan
