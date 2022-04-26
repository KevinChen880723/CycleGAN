import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self, num_residual_blocks, num_hourglass=8, use_variant=True):
        super().__init__()
        self.hourglass_list = nn.ModuleList()
        if not use_variant:
            print("Currently, using the original architecture to build the generator, num_hourglass will not be considered.")
        if use_variant:
            num_residual_blocks_array = np.ones(num_hourglass, dtype=int) * (num_residual_blocks // num_hourglass)
            num_residual_blocks_array[:num_residual_blocks % num_hourglass] += 1
        else:
            num_residual_blocks_array = np.array([num_residual_blocks])
        for num in num_residual_blocks_array:
            self.hourglass_list.append(HourGlass(num))

    def forward(self, x):
        for i, hourglass in enumerate(self.hourglass_list):
            if i == 0:
                hourglass_out, residual_out = hourglass(x, None)
            else:
                hourglass_out, residual_out = hourglass(hourglass_out, residual_out)
        return hourglass_out
    

class HourGlass(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(HourGlass, self).__init__()
        num_bottle_ch = 32

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
        self.downsampling_block = nn.Sequential(*sequential)

        sequential = list()
        # Residual blocks
        for i in range(num_residual_blocks):
            sequential += [ResidualBlock(num_bottle_ch*4)]
        self.residual_blocks = nn.Sequential(*sequential)
        
        # Upsampling
        sequential = list()
        sequential += [nn.ConvTranspose2d(num_bottle_ch*4, num_bottle_ch*2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(num_bottle_ch*2),
                       nn.ReLU(inplace=True)]
        sequential += [nn.ConvTranspose2d(num_bottle_ch*2, num_bottle_ch, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(num_bottle_ch),
                       nn.ReLU(inplace=True)]
        self.upsampling_block = nn.Sequential(*sequential) 

        # Output layer
        self.output_block = nn.Sequential(nn.ReflectionPad2d(3),
                                          nn.Conv2d(num_bottle_ch, 3, 7),
                                          nn.Tanh())

    def forward(self, x, residual_input=None):
        temp = self.downsampling_block(x)
        if residual_input != None:
            temp = temp + residual_input
        residual_out = self.residual_blocks(temp)
        temp = self.upsampling_block(residual_out)
        hourglass_out = self.output_block(temp)
        return hourglass_out, residual_out


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
