import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorL2H(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=3):
        super(GeneratorL2H, self).__init__()
        in_features = 128
        # Initial convolution block       
        model = [   nn.ReflectionPad2d(1),
                    nn.Conv2d(input_nc, in_features, 3),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ]

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(4):
            # model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
            #             nn.InstanceNorm2d(out_features),
            #             nn.ReLU(inplace=True) ]
            model += [  nn.Conv2d(in_features, out_features, 7, stride=1, padding=3),
                        nn.Upsample(scale_factor=2.0, mode='nearest'),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Tanh() 
                 ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class GeneratorH2L(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=3):
        super(GeneratorH2L, self).__init__()
        in_features = 8
        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        out_features = in_features*2
        for _ in range(4):
            model += [  nn.Conv2d(in_features, out_features, 7, stride=2, padding=3),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Output layer
        model += [  nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, output_nc, 3),
                    nn.Tanh() 
                 ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DiscriminatorH(nn.Module):
    def __init__(self, input_nc):
        super(DiscriminatorH, self).__init__()
        in_features = 8
        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, in_features, 7, stride=2, padding=3),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features, in_features*2, 7, stride=2, padding=3),
                    nn.InstanceNorm2d(in_features*2), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*2, in_features*4, 7, stride=2, padding=3),
                    nn.InstanceNorm2d(in_features*4), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*4, in_features*8, 7, stride=2, padding=3),
                    nn.InstanceNorm2d(in_features*8), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*8, in_features*16, 7, padding=3),
                    nn.InstanceNorm2d(in_features*16), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(in_features*16, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class DiscriminatorL(nn.Module):
    def __init__(self, input_nc):
        super(DiscriminatorL, self).__init__()
        in_features = 8
        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, in_features, 3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features, in_features*2, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(in_features*2), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*2, in_features*4, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(in_features*4), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*4, in_features*8, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(in_features*8), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_features*8, in_features*16, 3, padding=1),
                    nn.InstanceNorm2d(in_features*16), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(in_features*16, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)