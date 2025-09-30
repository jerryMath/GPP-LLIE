import torch
import torch.nn as nn

# Input tensor: batch=1, channels=2, size=3x3
x = torch.arange(18).float().view(1, 2, 3, 3)
print("Input:\n", x)

conv_normal = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, groups=1, bias=False)
print("Normal conv weight shape:", conv_normal.weight.shape)

conv_depthwise = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, groups=2, bias=False)
print("Depthwise conv weight shape:", conv_depthwise.weight.shape)


out_normal = conv_normal(x)
out_depthwise = conv_depthwise(x)

print("Normal conv output:\n", out_normal)
print("Depthwise conv output:\n", out_depthwise)

"""
weight.shape = [out_channels, in_channels/groups, kH, kW]
out_channels: num of filters
in_channels/groups: depth per filter


That’s why real models (like MobileNet) use:

# Depthwise (spatial filtering per channel)
dw = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, groups=C)

# Pointwise (mix channels after depthwise)
pw = nn.Conv2d(in_channels=C, out_channels=M, kernel_size=1)


Depthwise conv → learns per-channel spatial filters.

Pointwise conv → learns channel mixing (like a linear layer across channels).

This combination is called Depthwise Separable Convolution and is much cheaper than a full 3×3 conv.
"""
