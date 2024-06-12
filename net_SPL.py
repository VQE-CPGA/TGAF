import os
import mindspore.nn as nn
import mindspore
import numpy as np
import math
# import utils
# from ops.dcn.deform_conv import ModulatedDeformConv
from mindspore.ops import deformable_conv2d
from PIL import Image
import mindspore.nn as nn

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Cell):
    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True):
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, pad_mode='pad', padding=1, has_bias=False))
        modules_body.append(nn.ReLU()) 
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, pad_mode='pad', padding=1, has_bias=False))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.SequentialCell(*modules_body)

    def construct(self, x):
        res = self.body(x)
        res += x
        return res


## Channel Attention (CA) Layer
class CALayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.SequentialCell(
                nn.Conv2d(channel, channel, 1, padding=0, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 1, padding=0, has_bias=True),
                nn.Sigmoid()
        )

    def construct(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class Unet(nn.Cell):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(Unet, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        # self.deform_ks = deform_ks
        # self.size_dk = deform_ks ** 2
        # u-shape backbone
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.SequentialCell(
                    nn.Conv2d(nf, nf, base_ks, stride=2, pad_mode='pad', padding=base_ks//2),
                    nn.ReLU(),
                    nn.Conv2d(nf, nf, base_ks, pad_mode='pad', padding=base_ks//2),
                    nn.ReLU()
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.SequentialCell(
                    nn.Conv2d(2*nf, nf, base_ks, pad_mode='pad', padding=base_ks//2),
                    nn.ReLU(),
                    nn.Conv2dTranspose(nf, nf, 4, stride=2, pad_mode='pad', padding=1),
                    nn.ReLU()
                    )
                )
        self.tr_conv = nn.SequentialCell(
            nn.Conv2d(nf, nf, base_ks, stride=2, pad_mode='pad', padding=base_ks//2),
            nn.ReLU(),
            nn.Conv2d(nf, nf, base_ks, pad_mode='pad', padding=base_ks//2),
            nn.ReLU(),
            nn.Conv2dTranspose(nf, nf, 4, stride=2, pad_mode='pad', padding=1),
            nn.ReLU()
            )
        self.out_conv = nn.SequentialCell(
            nn.Conv2d(nf, nf, base_ks, pad_mode='pad', padding=base_ks//2),
            nn.ReLU()
            )


    def construct(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        # n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [inputs]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                mindspore.ops.cat([out, out_lst[i]], 1)
                )

        out = self.out_conv(out)


        return out



class Deform_block(nn.Cell):
    def __init__(self, in_nc, out_nc, nf, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(Deform_block, self).__init__()

        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, pad_mode='pad', padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        # self.deform_conv = ModulatedDeformConv(
        #     in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
        #     )
        # self.deform_conv = deformable_conv2d(in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc)

    def construct(self, inputs, feat):
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        # off_msk = self.offset_mask(self.out_conv(out))
        off_msk = self.offset_mask(feat)
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = mindspore.ops.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        # perform deformable convolutional fusion
        kh, kw = 1, 1
        deformable_groups = 1
        stride_h, stride_w = 1, 1
        dilation_h, dilation_w = 1, 1
        pad_h, pad_w = 0, 0
        x_h, x_w = 1, 2
        out_h = (x_h + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
        out_w = (x_w + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1
        # output = ops.deformable_conv2d(x, weight, offsets, (kh, kw), (1, 1, stride_h, stride_w,), (pad_h, pad_h, pad_w, pad_w), dilations=(1, 1, dilation_h, dilation_w))

        fused_feat = mindspore.ops.relu(mindspore.ops.deform_conv(inputs, off, msk))

        return fused_feat


# ==========
# Quality enhancement module
# ==========

class PlainCNN(nn.Cell):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        super(PlainCNN, self).__init__()

        self.in_conv = nn.SequentialCell(
            nn.Conv2d(in_nc, nf, base_ks, pad_mode='pad', padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(nf, nf, base_ks, pad_mode='pad', padding=1),
            nn.LeakyReLU(),
            )
        hid_conv_lst = []
        for _ in range(nb - 2):
            hid_conv_lst += [nn.Conv2d(nf, nf, base_ks, pad_mode='pad',padding=1),
                            nn.LeakyReLU(),]
            hid_conv_lst+=[RCB(nf)]
        self.hid_conv = nn.SequentialCell(*hid_conv_lst)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, pad_mode='pad', padding=1)

    def construct(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out



class ContextBlock(nn.Cell):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, has_bias=False)
        self.softmax = nn.Softmax(axis=2)

        self.channel_add_conv = nn.SequentialCell(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, has_bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, has_bias=False))

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = mindspore.ops.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def construct(self, x):
        # [N, C, 1, 1]
        # x = self.channel_add_conv(x)
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term
        # x = self.channel_add_conv(x)

        return x


### --------- Residual Context Block (RCB) ----------
class RCB(nn.Cell):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()
        act = nn.LeakyReLU()
        self.act = act
        nDiff = 16
        nFeat_slice = 4
        self.channel1 = n_feat//2
        self.channel2 = n_feat-self.channel1
        
        self.gcnet = ContextBlock(self.channel1, bias=bias)
        self.tail = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False, group=groups)
        
    def construct(self, x):
        x1, x2 = mindspore.ops.split(x,[self.channel1,self.channel2],dim=1)
        res1 = self.act(self.gcnet(x1))
        com1 = res1 + x2
        res2 = self.act(self.gcnet(com1))
        com2 = res2 + com1
        res = self.tail(mindspore.ops.cat((com1,com2),dim=1))

        return res



class STDF_simple(nn.Cell):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF_simple, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # u-shape backbone
        self.in_conv = nn.SequentialCell(
            nn.Conv2d(in_nc, nf, base_ks, pad_mode='pad',padding=base_ks//2),
            nn.LeakyReLU()
            )

        
        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, pad_mode='pad', padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        # self.deform_conv = ModulatedDeformConv(
        #     in_nc, out_nc, deform_ks, pad_mode='pad', padding=deform_ks//2, deformable_groups=in_nc
        #     )

    def construct(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        inputs = self.in_conv(inputs)

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask((inputs)) # self.out_conv
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = mindspore.ops.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        # perform deformable convolutional fusion
        # mindspore.ops.deform_conv(inputs, off, msk)
        # fused_feat = mindspore.ops.relu(self.deform_conv(inputs, off, msk))
        fused_feat = mindspore.ops.relu(mindspore.ops.deform_conv(inputs, off, msk))

        return fused_feat

# ==========
# TGDA network
# ==========
class TGDA_7(nn.Cell):
    def __init__(self):
        super(TGDA_7,self).__init__()
        self.radius = 3
        self.input_len = 2 * self.radius + 1
        self.center = self.input_len // 2
        self.color = 1 

        self.inputconv = nn.SequentialCell(nn.Conv2d(in_channels=self.radius, out_channels=64, kernel_size=3, pad_mode='pad', padding=3//2),
            nn.LeakyReLU())
        self.Unet = Unet(
            in_nc=64,   
            out_nc=64, 
            nf=64, 
            nb=3, 
            base_ks = 3,
            )


        self.deform_block_1 = Deform_block(
            in_nc=1 * self.radius,  
            out_nc=64, 
            nf=64, 
            base_ks = 3,
            deform_ks= 1 
        )

        self.deform_block_3 = Deform_block(
            in_nc=1 * self.radius,  
            out_nc=64, 
            nf=64, 
            base_ks = 3,
            deform_ks= 3 
        )

        self.deform_block_5 = Deform_block(
            in_nc=1 * self.radius,  
            out_nc=64, 
            nf=64, 
            base_ks = 3,
            deform_ks= 5 
        )       

      
        self.RCAB = RCAB(64)
        self.fuse = nn.SequentialCell(
            nn.Conv2d(64*2, 64, 3, stride=1, pad_mode='pad',padding=3//2),
            nn.LeakyReLU(),
        )

        self.wpnet = STDF_simple(
            in_nc=64, 
            out_nc=64, 
            nf=64, 
            nb=3, 
            base_ks = 3,
            deform_ks = 1
            )

        self.qenet = PlainCNN(
            in_nc=64,  
            nf=64, 
            nb=5, 
            out_nc=1,
            )

    # x is the input reference frames
    # y is the preceding hidden state feature
    def construct(self,x):
        x = x.contiguous()
        G_1 = x[:,self.center-1:self.center+2,:,:].contiguous()   #  self.center = 3
        G_2 = x[:,1::2,:,:].contiguous()
        G_3 = x[:,0::3,:,:].contiguous()
        
        # [B F H W]
        out_1 = self.Unet(self.inputconv(G_1))
        out_1 = self.deform_block_1(G_1, out_1)
        out_2 = self.Unet(self.inputconv(G_2))
        out_2 = self.deform_block_3(G_2, out_2)
        out_3 = self.Unet(self.inputconv(G_3))
        out_3 = self.deform_block_5(G_3, out_3)

        # [B 14 H W]
        # out = out_1 + out_2 + out_3
        out_f1 = self.fuse(mindspore.ops.cat((out_1,out_2),1))
        out_f2 = self.fuse(mindspore.ops.cat((out_2,out_3),1))
        out = self.RCAB(self.fuse(mindspore.ops.cat((out_f1,out_f2),1)))
        # [B F H W]
        out = self.wpnet(out)
        # N, C, H, W = out.shape

        final_out = self.qenet(out) + x[:, [self.radius + i*(2*self.radius+1) for i in range(self.color)], ...]
        return final_out


