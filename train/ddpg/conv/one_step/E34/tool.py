import model.utils as model_utils
from torch import nn
import ae

def load_ae_model():
	enc = model_utils.Model(4,
	                        [
		                        '1K5D1S1C32B1',
		                        '1K3D1S2C64B1',
		                        '1K3D1S1C64B1',
		                        '1K3D1S2C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D2S1C128B1',
		                        '1K3D4S1C128B1',
		                        '1K3D8S1C128B1',
		                        '1K3D16S1C128B1',
		                        '1K3D1S1C128B1',
		                        '1K3D1S1C128B1',
		                    ]
	                        )

	dec = model_utils.Model(128,
	                [
		                nn.Upsample((128,128)),
						'1K3D1S1C64B1',
						'1K3D1S1C64B1',
		                nn.Upsample((256,256)),
						'1K3D1S1C32B1',
						'1K3D1S1C16B1',
						'1K3D1S1C3B1',
		                nn.Tanh()
	                ])

	m = ae.model.AE(enc,dec)
	return m