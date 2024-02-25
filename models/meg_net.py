import torch
from torch import nn
from torch.nn import functional as F
import torchaudio as ta

from .common import (
    ConvSequence, ScaledEmbedding, SubjectLayers,
    DualPathRNN, ChannelMerger, ChannelDropout, pad_multiple
)

class SeparableConv1d(nn.Module):
    # https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
"""
def MEGNet(embed_dim = 768, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    #softmax      = Activation('softmax', name = 'softmax')(dense)

    # block3       = LayerNormalization()
    
    return Model(inputs=input1, outputs=softmax)

"""

class MEGNet(nn.Module):
    def __init__(self, input_ch = 272, output_size = 768, 
                dropout_rate=0.5, F1=4, F2=8, kernLength=64, # changed from F1=8, F2=16 original
                D = 2,):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=F1, 
                        kernel_size=(kernLength), padding=(kernLength // 2), bias=False)
        
        self.batchNorm1 = nn.BatchNorm1d(F1) # should this be 2d or 1d?
        
        self.depthwiseConv1 = nn.Conv1d(in_channels=F1, out_channels=F1 * D, # this needs checking
                                        kernel_size=(kernLength), groups=F1, padding=1, bias=False)

        self.batchNorm2 = nn.BatchNorm1d(F1 * D)

        self.elu = nn.ELU()

        self.avgPool = nn.AvgPool1d(kernel_size=(4))
      
        self.dropout = nn.Dropout1d(p=dropout_rate)

        self.separableConv = SeparableConv1d(in_channels=F2, out_channels=F2, kernel_size=(4), bias=False) # changed kernel size 16->4

        self.batchNorm = nn.BatchNorm1d(F2)

        self.elu2 = nn.ELU()

        self.avgPool2 = nn.AvgPool1d(kernel_size=(8))

        self.dropout2 = nn.Dropout1d(p=dropout_rate)

        self.flatten = nn.Flatten() # change to temporal aggregation

        self.mlp_proj = nn.Sequential(nn.LayerNorm(8), nn.GELU(), nn.Linear(8,output_size)) # why 2048? changed to 8

    def forward(self, x):
        print(f"Incoming shape is {x.shape}")
        x = self.conv1(x)
        print(f"Shape after convolution is {x.shape}")
        x = self.batchNorm1(x)
        print(f"Shape after batchnorm1 is {x.shape}")
        # x = self.elu(x)
        # print(f"Shape after elu is {x.shape}")
        x = self.depthwiseConv1(x)
        print(f"Shape after depthwiseconv is {x.shape}")
        x = self.batchNorm2(x)
        print(f"Shape after batchnorm2 is {x.shape}")
        x = self.elu(x)
        print(f"Shape after elu is {x.shape}")
        x = self.avgPool(x)
        print(f"Shape after avgpool is {x.shape}")
        x = self.dropout(x)
        print(f"Shape after dropout is {x.shape}")
        x = self.separableConv(x)
        print(f"Output shape after separable conv is {x.shape}")
        x = self.batchNorm(x)
        print(f"Output shape after batchnorm is {x.shape}")
        x = self.elu2(x)
        print(f"Output shape after elu2 is {x.shape}")
        x = self.avgPool2(x)
        print(f"Output shape after avgpool2 is {x.shape}")
        x = self.dropout2(x)
        print(f"Output shape after dropout2 is {x.shape}")
        x = self.flatten(x)
        print(f"Output shape after flatten is {x.shape}")
        x = self.mlp_proj(x)
        print(f"Output shape after mlp_proj is {x.shape}")
        return x
