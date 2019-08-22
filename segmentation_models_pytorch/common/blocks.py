import torch.nn as nn
import torch


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_parmas):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, bias=not (use_batchnorm))
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, **batchnorm_parmas))
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    

class sSE(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = ConvBn2d(in_channels=out_channels, out_channels=1, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x
    
    
class cSE(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_features=out_channels, out_features=int(out_channels / 2), bias=False)
        self.linear2 = nn.Linear(in_features=int(out_channels / 2), out_features=out_channels, bias=False)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
        y = self.linear1(y)
        y = torch.relu(y)
        y = self.linear2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)