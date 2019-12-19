from torch import nn

class Encoder(nn.Module):
    def __init__(self, enc,dim_z):
        super(Encoder, self).__init__()
        self.m = enc(dim_z)
        self.dim_z = dim_z

    def forward(self, x):
        return self.m(x).chunk(2, dim=1)

class Decoder(nn.Module):
    def __init__(self, dec, dim_z):
        super(Decoder, self).__init__()
        self.m = dec(dim_z)
        self.dim_z=dim_z

    def forward(self, x):
        return self.m(x)

class FaceEncoder(nn.Module):
    def __init__(self, dim_z):
        super(FaceEncoder, self).__init__()

        self.dim_z = dim_z

        self.m = nn.Sequential(
            #todo 3*64*64
            nn.Conv2d(3,64,5,2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #todo 64*32*32
            nn.Conv2d(64, 64*2, 5, 2, 2),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            #todo 128*16*16
            nn.Conv2d(64*2, 64*4, 5, 2, 2),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            #todo 256*8*8
        )

        self.out_reshape = nn.Sequential(
            nn.Linear(64*4*8*8,2*dim_z),
            nn.BatchNorm1d(2*dim_z),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.m(x)
        x = self.out_reshape(x.view(-1,64*4*8*8))
        return x

class FaceDecoder(nn.Module):
    def __init__(self, dim_z):
        super(FaceDecoder, self).__init__()

        self.dim_z = dim_z
        self.m = nn.Sequential(
            nn.ConvTranspose2d(64*4,64*4,5,1,2),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            #8x8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            #16x16
            nn.ConvTranspose2d(64 * 2, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #32
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
            #64
        )
        self.in_reshape = nn.Sequential(
            nn.Linear(dim_z,64*4*8*8),
            nn.BatchNorm1d(64*4*8*8),
            nn.ReLU(),
        )


    def forward(self, x):
        x = self.in_reshape(x).view(-1,64*4,8,8)
        x = self.m(x)
        return x

_CONFIG_MAP = {
    'vae':(FaceEncoder,FaceDecoder),
}

def load_config(name):
    """Load a particular configuration
    Returns:
    (encoder, transition, decoder) A tuple containing class constructors
    """
    if name not in _CONFIG_MAP.keys():
        raise ValueError("Unknown config: %s", name)
    return _CONFIG_MAP[name]

__all__ = ['load_config']
