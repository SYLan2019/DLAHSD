import math
import torch.nn as nn
from Attention.siamAM import simam_module
from Attention.SE import se_block
class ISA_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(ISA_Attention, self).__init__()


        self.channel_attention = se_block(in_channels)


        self.siam = simam_module()

    def forward(self, x):

        out = self.siam(self.channel_attention(x))


        return out


