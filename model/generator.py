import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, num_hourglass=8, use_variant=True):
        super().__init__()
        sequence = list()
        if not use_variant:
            print("Currently, using the original architecture to build the generator, num_hourglass will not be considered.")
        for i in range(num_hourglass if use_variant else 1):
            sequence.append(HourGlass(num_hourglass/2 if use_variant else 1, use_variant))
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
    

class HourGlass(nn.Module):
    def __init__(self, ch_drop_rate=4, use_variant=True):
        super(HourGlass, self).__init__()
        assert ch_drop_rate <= 8, 'ch_drop_rate should <= 8'
        num_bottle_ch = int(64 // ch_drop_rate) if use_variant else 64

        # Initial convolution block
        sequential = [nn.ReflectionPad2d(3),
                      nn.Conv2d(3, num_bottle_ch, 7),
                      nn.InstanceNorm2d(num_bottle_ch),
                      nn.ReLU(inplace=True)]

        # Downsampling
        sequential += [nn.Conv2d(num_bottle_ch, num_bottle_ch*2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(num_bottle_ch*2),
                       nn.ReLU(inplace=True)]
        sequential+= [nn.Conv2d(num_bottle_ch*2, num_bottle_ch*4, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(num_bottle_ch*4),
                      nn.ReLU(inplace=True)]

        # Residual blocks
        for i in range(4 if use_variant else 9):
            sequential += [ResidualBlock(num_bottle_ch*4)]

        # Upsampling
        sequential += [nn.ConvTranspose2d(num_bottle_ch*4, num_bottle_ch*2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(num_bottle_ch*2),
                       nn.ReLU(inplace=True)]
        sequential += [nn.ConvTranspose2d(num_bottle_ch*2, num_bottle_ch, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(num_bottle_ch),
                       nn.ReLU(inplace=True)]

        # Output layer
        sequential += [nn.ReflectionPad2d(3),
                       nn.Conv2d(num_bottle_ch, 3, 7),
                       nn.Tanh()]
        self.sequence = nn.Sequential(*sequential)

    def forward(self, x):
        return self.sequence(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        sequential = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, in_channels, 3),
                      nn.InstanceNorm2d(in_channels),
                      nn.ReLU(inplace=True)]

        sequential += [nn.ReflectionPad2d(1),
                       nn.Conv2d(in_channels, in_channels, 3),
                       nn.InstanceNorm2d(in_channels)] 
        
        self.residual = nn.Sequential(*sequential)

    def forward(self, x):
        return x + self.residual(x)
