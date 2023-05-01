"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

# now write your custom layer
class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(CustomGroupedConv2D, self).__init__()
        self.groups = groups
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels//groups, out_channels=out_channels//groups, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias) for _ in range(groups)])

    def forward(self, x):
        x_split = torch.split(x, split_size_or_sections=int(x.shape[1] / self.groups), dim=1)
        out = torch.cat([self.convs[i](x_split[i]) for i in range(self.groups)], dim=1)
        return out
    
# custom grouped 2d convolution
custom_grouped_layer = CustomGroupedConv2D(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# copy weights and bias from original layer to custom layer
for i in range(16):
    custom_grouped_layer.convs[i].weight.data = w_torch[i*8:(i+1)*8,:,:,:].clone()
    custom_grouped_layer.convs[i].bias.data = b_torch[i*8:(i+1)*8].clone()
    
y_custom = custom_grouped_layer(x)

# check if the outputs are equal
print(torch.allclose(y, y_custom))


# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
