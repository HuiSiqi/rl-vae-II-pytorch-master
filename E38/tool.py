import model.utils as model_utils,torch
from torch import nn
import vae_wgan_gp,ae
def load_vae_model():
	enc = model_utils.Model(3,
	                        [
		                        '1K5D1S1C32B1',
		                        '1K3D1S2C64B1',
		                        '1K3D1S1C64B1',
		                        '1K3D1S2C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D1S2C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D2S1C128B1',
		                        '1K3D4S1C128B1',
		                        '1K3D8S1C128B1',
		                        '1K3D16S1C128B1',
		                        '1K3D1S1C64B1',
		                        '1K3D1S1C16B1',
		                    ]
	                        )

	dec = model_utils.Model(8,
	                [
		                '0K4D1S2C256B1',
		                '1K3D1S1C256B1',
		                '0K4D1S2C128B1',
		                '1K3D1S1C128B1',
		                '1K3D1S1C64B1',
		                '0K4D1S2C3B1',
		                nn.Tanh()
	                ])

	m = vae_wgan_gp.model.VAE(enc,dec)
	return m

# def load_ae_model():
# 	enc = model_utils.Model(3,
# 	                        [
# 		                        '1K5D1S1C32B1',
# 		                        '1K3D1S2C64B1',
# 		                        '1K3D1S1C64B1',
# 		                        '1K3D1S2C128B1',
# 		                        '1K3D1S1C128B1',
# 		                        '1K3D1S2C128B1',
# 		                        '1K3D1S1C128B1',
# 		                        '1K3D2S1C128B1',
# 		                        '1K3D4S1C128B1',
# 		                        '1K3D8S1C128B1',
# 		                        '1K3D16S1C128B1',
# 		                        '1K3D1S1C64B1',
# 		                        '1K3D1S1C8B1',
# 		                    ]
# 	                        )
#
# 	dec = model_utils.Model(8,
# 	                [
# 		                '0K4D1S2C256B1',
# 		                '1K3D1S1C256B1',
# 		                '0K4D1S2C128B1',
# 		                '1K3D1S1C128B1',
# 		                '1K3D1S1C64B1',
# 		                '0K4D1S2C3B1',
# 		                nn.Tanh()
# 	                ])
#
# 	m = ae.model.AE(enc,dec)
# 	return m

def load_ae_model():
	enc = model_utils.Model(3,
	                        [
		                        '1K5D1S1C32B1',
		                        '1K3D1S2C64B1',
		                        '1K3D1S1C64B1',
		                        '1K3D1S2C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D1S2C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D2S1C128B1',
		                        '1K3D4S1C128B1',
		                        '1K3D8S1C128B1',
		                        '1K3D16S1C128B1',
		                        # '1K3D1S1C64B1',
		                        # '1K3D1S1C8B1',
		                    ]
	                        )

	dec = model_utils.Model(128,
	                [
		                '0K4D1S2C256B1',
		                '1K3D1S1C256B1',
		                '0K4D1S2C128B1',
		                '1K3D1S1C128B1',
		                '1K3D1S1C64B1',
		                '0K4D1S2C3B1',
		                nn.Tanh()
	                ])

	m = ae.model.AE(enc,dec)
	return m

def load_discriminator():
	return model_utils.Model(3,config=[
		'1K5D1S2C64B1',
		'1K5D1S2C128B1',
		'1K5D1S2C256B1',
		'1K5D1S2C256B1',
		'1K5D1S2C256B1',
		'1K8D1S0C1B0',
	])


def load_regressor():
	model = model_utils.Model(3,
	                [
		                '1K5D1S1C32B1',
                        '1K3D1S2C64B1',
                        '1K3D1S1C64B1',
                        '1K3D1S2C128B1',
                        '1K3D1S1C128B1',
                        '1K3D1S2C128B1',
                        '1K3D1S1C128B1',
                        '1K3D2S1C128B1',
                        '1K3D4S1C128B1',
                        '1K3D8S1C128B1',
                        '1K3D16S1C128B1',
		                '0K4D1S2C256B1',
		                '1K3D1S1C256B1',
		                '0K4D1S2C128B1',
		                '1K3D1S1C128B1',
		                '1K3D1S1C64B1',
		                '0K4D1S2C3B1',
	                ])
	return model




