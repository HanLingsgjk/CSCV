import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.submodule import  bfmodule
from core.utils.gma import Aggregate
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
class expHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(expHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.tanh(self.conv2(self.relu(self.conv1(x))))
class expHeadm(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(expHeadm, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
class deprhHead(nn.Module):#这个用来修正深度
    def __init__(self, input_dim=128, hidden_dim=256):
        super(deprhHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
class bmaskHead(nn.Module):#这个用来提供分类掩膜
    def __init__(self, input_dim=128, hidden_dim=256):
        super(bmaskHead, self).__init__()
        self.mask_unit = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
                                      nn.BatchNorm2d(int(hidden_dim)),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(hidden_dim, 32, 1))
    def forward(self, x):
        return self.mask_unit(x)


class occHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(expHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv2(self.relu(self.conv1(x))))
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
class ConvGRUd(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128, dilation=4):
        super(ConvGRUd, self).__init__()
        self.hidden_dim = hidden_dim
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, dilation=dilation, padding=dilation)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz1(hx) + self.convz2(hx))
        r = torch.sigmoid(self.convr1(hx) + self.convr2(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)) + self.convq2(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class dcSmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(dcSmallMotionEncoder, self).__init__()
        cor_planes = 3 * (2*3 + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 32, 7, padding=3)
        self.conve2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv = nn.Conv2d(128+32, 80, 3, padding=1)

    def forward(self, flow, corr ,exp):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        cor_flo = torch.cat([cor, flo ,expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
class BasicMotioneEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoder, self).__init__()
        cor_planes = 343#411
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = nn.Conv2d(64+192+64, 128-3, 3, padding=1)

    def forward(self, flow, corr, exp):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        cor_flo = torch.cat([cor, flo,expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp], dim=1)
class BasicMotioneEncoder_depth(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoder_depth, self).__init__()
        cor_planes = 343#args.corr_levels * (2*args.corr_radius + 1)**2#343#223
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convd2 = nn.Conv2d(128, 32, 3, padding=1)

        self.conv = nn.Conv2d(64+192+64+32, 128-3-1, 3, padding=1)

    def forward(self, flow, corr, exp,d1):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        d1o = F.relu(self.convd1(d1))
        d1o = F.relu(self.convd2(d1o))

        cor_flo = torch.cat([cor, flo,expo,d1o], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp,d1], dim=1)
class BasicMotioneEncoder_split_scale(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoder_split_scale, self).__init__()
        cor_planes = 147#args.corr_levels * (2*args.corr_radius + 1)**2#343#223
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(1, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192+64, 128-3, 3, padding=1)

    def forward(self, flow, corr, exp):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        cor_flo = torch.cat([cor, flo,expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp], dim=1)
class BasicMotioneEncoder_split(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoder_split, self).__init__()
        cor_planes = 196#args.corr_levels * (2*args.corr_radius + 1)**2#343#223
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)#128
class BasicMotioneEncoder2(nn.Module):
    def __init__(self, args):
        super(BasicMotioneEncoder2, self).__init__()
        cor_planes = 294
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(24, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conve1 = nn.Conv2d(13, 128, 7, padding=3)
        self.conve2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192+64, 128-3, 3, padding=1)

    def forward(self, flow, corr, exp):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        expo = F.relu(self.conve1(exp))
        expo = F.relu(self.conve2(expo))

        cor_flo = torch.cat([cor, flo,expo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow, exp], dim=1)
class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow
class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoderorin(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim + 128)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=128)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)

    def forward(self, net, inp, corr, flow,exp, attention):
        motion_features = self.encoder(flow, corr, exp)
        motion_features_global = self.aggregator(attention, motion_features)
        inp = torch.cat([inp, motion_features, motion_features_global], dim=1)# 128+192


        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow, exp_flow
class BasicdcUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(BasicdcUpdateBlock, self).__init__()
        self.args = args
        self.encoder = dcSmallMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=96, input_dim=146)
        self.flow_head = FlowHead(96, hidden_dim=256)
        self.exp_head = expHead(96, hidden_dim=128)
        self.conbine = nn.Sequential(
            nn.Conv2d(128+96, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 96, 1, padding=0))
    def forward(self, net ,dnet, inp, corr, flow,exp, upsample=True):
        motion_features = self.encoder(flow, corr ,exp)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.conbine(torch.cat([net, dnet], dim=1))
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)

        return net, delta_flow, exp_flow
class BasicpyUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(BasicpyUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head  = expHead(hidden_dim, hidden_dim=128)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.exp_head = expHead(hidden_dim, hidden_dim=256)
    def forward(self, net, inp, corr, flow, exp, upsample=True):
        motion_features = self.encoder(flow, corr, exp)#128
        inp = torch.cat([inp, motion_features], dim=1)#128+192

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow, exp_flow
class ScaleflowUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        # self.gru = ConvGRUd(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=256)
        #self.occ_head = occHead(hidden_dim, hidden_dim=128)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp, upsample=True):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class ScaleflowUpdateBlock_bkgo(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlock_bkgo, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        # self.gru = ConvGRUd(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHeadm(hidden_dim, hidden_dim=256)
        #self.occ_head = occHead(hidden_dim, hidden_dim=128)
        self.bkmask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1, padding=0))

        self.fusion = nn.Sequential(
            nn.Conv2d(128*2+2, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
    def forward(self, net, inp, corr, flow, exp, corr_bk, flow_prebk,exp_bk,bk_mask,mid_pre, upsample=True):
        motion_featurea = self.encoder(flow, corr, exp)  # 128
        motion_featureb = self.encoder(flow_prebk, corr_bk, exp_bk)  # 128
        motion_cat = torch.cat([motion_featurea,motion_featureb,bk_mask,mid_pre],dim=1)
        motion_features = self.fusion(motion_cat)

        inp = torch.cat([inp, motion_features], dim=1)  # 128+192

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)

        bk_mask = self.bkmask(net)
        bk_mask = torch.nn.Sigmoid()(bk_mask)
        # scale mask to balence gradients
        masks = .25 * self.masks(net)
        return net,masks,bk_mask, delta_flow, exp_flow


class ScaleflowUpdateBlockbk(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlockbk, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.encoderbk = BasicMotioneEncoder(args)

        self.flow_head = FlowHead(512, hidden_dim=256)
        self.exp_head = expHead(512, hidden_dim=256)
        #self.occ_head = occHead(hidden_dim, hidden_dim=128)
        self.fusion = nn.Sequential(
            nn.Conv2d(128*2+1, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1, padding=0))

        self.mask = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))


    def forward(self, net, inp, corr_e, corr_bk, flow, flow_prebk, exp, exp_bk, bk_mask, upsample=True):
        motion_features = self.encoder(flow, corr_e, exp)  # 128
        motion_featuresbk = self.encoderbk(flow_prebk, corr_bk*bk_mask, exp_bk)  # 128
        motion_fusion = self.fusion(torch.cat([motion_features,motion_featuresbk,bk_mask],dim=1))
        net = torch.cat([inp, motion_fusion,net], dim=1)


        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return  mask,masks, delta_flow, exp_flow
class ScaleflowmaskUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowmaskUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        # self.gru = ConvGRUd(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHeadm(hidden_dim, hidden_dim=128)
        #self.bmask_head = bmaskHead(hidden_dim, hidden_dim=128) #这个是给刚性语义层编码

        #self.occ_head = occHead(hidden_dim, hidden_dim=128)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        #这个应该考虑
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))


    def forward(self, net, inp, corr, flow, exp, upsample=True):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192

        net = self.gru(net, inp)
        #一个隐含的语义层

        #ae  = self.bmask_head(net)
        #给出深度的残差项，这个是修正第一帧的深度

        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)#这个和深度上采样共用
        return net, mask,masks, delta_flow, exp_flow
class Scale_split_flowUpdateBlock2(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(Scale_split_flowUpdateBlock2, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder_split_scale(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.exp_head = expHead(hidden_dim, hidden_dim=128)
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, exp, upsample=True):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192

        net = self.gru(net, inp)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        masks = .25 * self.masks(net)
        return net,masks, exp_flow
class Scale_split_flowUpdateBlock1(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(Scale_split_flowUpdateBlock1, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder_split(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)

        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
class ScaleflowUpdateBlock_scale(nn.Module):
    def __init__(self, args, hidden_dim=96, input_dim=128):
        super(ScaleflowUpdateBlock_scale, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
       # self.gru = ConvGRUd(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.exp_head = expHead(hidden_dim, hidden_dim=128)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.masks = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        self.exp_head = expHead(hidden_dim, hidden_dim=256)

    def forward(self, net, inp, corr, flow, exp, upsample=True):
        motion_features = self.encoder(flow, corr, exp)  # 128
        inp = torch.cat([inp, motion_features], dim=1)  # 128+192

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        exp_flow = self.exp_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        masks = .25 * self.masks(net)
        return net, mask,masks, delta_flow, exp_flow
class DCUpdateBlockb(nn.Module):
    def __init__(self, args):
        super(DCUpdateBlockb, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)

        self.mask = nn.Sequential(
            nn.Conv2d(498, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = bfmodule(498, 1)


    def forward(self,dnet, dinp, net, inp, corr, flow,exp):
        motion_features = self.encoder(flow, corr, exp)
        inp = torch.cat([inp, motion_features,net,dnet,dinp], dim=1)


        exp_flow = self.exp_head(inp)
        # scale mask to balence gradients
        mask = .25 * self.mask(inp)
        return mask ,exp_flow[0]
class DCUpdateBlock(nn.Module):
    def __init__(self, args,hidim):
        super(DCUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)

        self.mask = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = bfmodule(128+hidim*2, 1)
    def forward(self, net, inp, corr, flow,exp):
        motion_features = self.encoder(flow, corr, exp)
        inp = torch.cat([inp, motion_features,net], dim=1)

        exp_flow = self.exp_head(inp)
        # scale mask to balence gradients
        mask = .25 * self.mask(net.float())
        return mask ,exp_flow[0]
class DCUpdateBlock_mask(nn.Module):
    def __init__(self, args,hidim):
        super(DCUpdateBlock_mask, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)

        self.mask = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = bfmodule(128+hidim*2, 1)
        self.bkmask_head = bfmodule(128 + hidim * 2, 1)
    def forward(self, net, inp, corr, flow,exp):
        motion_features = self.encoder(flow, corr, exp)
        inp = torch.cat([inp, motion_features,net], dim=1)

        exp_flow = self.exp_head(inp)
        bk_mask = self.bkmask_head(inp.detach())[0]
        bk_mask = torch.nn.Sigmoid()(bk_mask)
        # scale mask to balence gradients
        mask = .25 * self.mask(net.float())
        return mask ,exp_flow[0],bk_mask
class DCUpdateBlock_mask_dc(nn.Module):
    def __init__(self, args,hidim):
        super(DCUpdateBlock_mask_dc, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder_depth(args)

        self.mask = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = bfmodule(128+hidim*2, 1)
    def forward(self, net, inp, corr, flow,exp,depth1):
        motion_features = self.encoder(flow, corr, exp,depth1)
        inp = torch.cat([inp, motion_features,net], dim=1)

        exp_flow = self.exp_head(inp)
        # scale mask to balence gradients
        mask = .25 * self.mask(net.float())
        return mask ,exp_flow[0]
#遮挡估计模块
#输入有net
class OCCpdateBlock(nn.Module):
    def __init__(self, args,hidim):
        super(OCCpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)

        self.mask = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.occ_head = bfmodule(128+hidim*2, 1)
    def forward(self, net, inp, corr, flow_un,exp_un):
        motion_features = self.encoder(flow_un, corr, exp_un)
        inp = torch.cat([inp, motion_features,net], dim=1)

        occ_flow = self.occ_head(inp)
        # scale mask to balence gradients
        mask = .25 * self.mask(net.float())
        return mask ,occ_flow[0]
class DCUpdateBlock_split(nn.Module):
    def __init__(self, args,hidim):
        super(DCUpdateBlock_split, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder_split_scale(args)

        self.mask = nn.Sequential(
            nn.Conv2d(hidim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.exp_head = bfmodule(128+hidim*2, 1)
    def forward(self, net, inp, corr, flow, exp):
        motion_features = self.encoder(flow, corr, exp)
        inp = torch.cat([inp, motion_features,net], dim=1)

        exp_flow = self.exp_head(inp)
        # scale mask to balence gradients
        mask = .25 * self.mask(net.float())
        return mask ,exp_flow[0]
class DCUpdateBlockpy(nn.Module):
    def __init__(self, args):
        super(DCUpdateBlockpy, self).__init__()
        self.args = args
        self.encoder = BasicMotioneEncoder(args)

        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9*2, 1, padding=0))
        self.exp_head = bfmodule(512, 2)


    def forward(self, net, inp, corr, flow,exp):
        motion_features = self.encoder(flow, corr, exp)
        inp = torch.cat([inp, motion_features,net], dim=1)


        exp_flow = self.exp_head(inp)
        # scale mask to balence gradients
        mask = .25 * self.mask(net.float())
        return mask ,exp_flow[0]