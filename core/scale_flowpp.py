#ResScale_TDSKGOOD_Kft.pth
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.extractor import ResNetFPN
from core.corr import CorrpyBlockResScale,CorrBlock
from core.update import ResFlowUpdateBlockUnetL,DCFlowUpdateBlock
from core.utils.submodule import Tiny_Unetlikev7,Tiny_Unetlikev2
from utils.utils import  coords_grid
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class RGBHead(nn.Module):
    def __init__(self, input_dim=128):
        super(RGBHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 3, 3, padding=1)
        self.relu = nn.Tanh()

    def forward(self, x):
        return self.relu(self.conv1(x))
# best_weight raft-kitti_11.pth
def gaussian2D2(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h/h.sum()

from core.layer import ConvNeXtV2tiny
class ResScale_AB(nn.Module):
    def __init__(self, args):
        super(ResScale_AB, self).__init__()
        args.dim = 192
        self.args = args
        # feature network, context network, and update block
        self.fnet = ResNetFPN(args, input_dim=3, output_dim=256, norm_layer=nn.InstanceNorm2d, init_weight=True,pretrain = 'resnet18')
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True,pretrain = 'resnet18')
        self.init_conv = nn.Conv2d(2 * args.dim,  2 * args.dim, kernel_size=3, stride=1, padding=1)
        #在预测光流之前加一个ConvNextX模块
        self.init_flowconv = Tiny_Unetlikev7(722, args.dim)#594

        self.flow_head16x = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, args.dim, 3, padding=1)
        )
        self.exp_head16x = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, args.dim, 3, padding=1)
        )
        self.flow_head8x = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1)
        )
        self.exp_head8x = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.upsample_weightexp = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.hidden_dim = hdim = args.dim
        self.update_block = ResFlowUpdateBlockUnetL(self.args, hidden_dim=hdim)

        self.update_blockUnet = DCFlowUpdateBlock(self.args,hdim)
        xita = 2 ** 0.25  # 0.25 + 1
        self.delta1 = 1.75 / 2
        kernel1 = gaussian2D2([5, 5], sigma=(xita, xita))
        xita = 2 ** 0.5  # 0.5 + 1
        self.delta2 = 1.5 / 2
        kernel2 = gaussian2D2([5, 5], sigma=(xita, xita))
        xita = 2 ** 0.75  # 0.75 + 1
        self.delta3 = 1.25 / 2
        kernel3 = gaussian2D2([5, 5], sigma=(xita, xita))
        xita = 2 ** 1  # 1 + 1
        self.delta4 = 1 / 2
        kernel4 = gaussian2D2([5, 5], sigma=(xita, xita))

        kernel = torch.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weightt1 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weightt2 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weightt3 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel4).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weightt4 = nn.Parameter(data=kernel, requires_grad=False)

        self.mseloss = nn.MSELoss(reduction='mean')

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def initialize_exp(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape

        exp = torch.ones((N, H // 8, W // 8)).to(img.device) * 2

        # optical flow computed as difference: flow = coords1 - coords0
        return exp
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)
    def upsample_exp(self, exp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = exp.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        m = nn.ReplicationPad2d(1)
        up_exp = F.unfold(m(exp), [3, 3])
        up_exp = up_exp.view(N, 1, 9, 1, 1, H, W)

        up_exp = torch.sum(mask * up_exp, dim=2)
        up_exp = up_exp.permute(0, 1, 4, 2, 5, 3)
        return up_exp.reshape(N, 1, 8 * H, 8 * W)
    def change_fun(self, exp):  # 将坐标转换成膨胀率
        exp = exp.clamp(0, 4)
        x = exp * 0.25 + 0.5
        return x
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    def initialize_flow16x(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 16, W // 16).to(img.device)
        coords1 = coords_grid(N, H // 16, W // 16).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    def forward(self, image1, image2, iters=6, test_mode=False):
        """ Estimate optical flow between pair of frames """
        _, _, h, w = image1.shape
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        #对图片进行多尺度采样
        w1 = self.weightt4
        image21 = F.conv2d(image2, w1, padding=2, groups=3)
        image21 = F.interpolate(image21, [int(h * self.delta4), int(w * self.delta4)])
        w2 = self.weightt2
        image23 = F.conv2d(image2, w2, padding=2, groups=3)
        image23 = F.interpolate(image23, [int(h * self.delta2), int(w * self.delta2)])
        image26 = F.interpolate(image2, [int(h * 1.25), int(w * 1.25)])
        image28 = F.interpolate(image2, [int(h * 1.5), int(w * 1.5)])
        flow_predictions = []
        exp_predictions = []
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        with autocast(enabled=self.args.mixed_precision):
            #首先是特征层推理预先预测光流
            cnet,cnet16x = self.cnet(torch.cat([image1, image2], dim=1))
            cnet = self.init_conv(cnet)
            net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        Fmap1 = []
        Fmap2 = []
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1,fmap1u = self.fnet(image1)
            Fmap1.append(fmap1.float())

            fmap2 = self.fnet(image21,ifonly1=True)
            Fmap2.append(fmap2.float())

            fmap2 = self.fnet(image23,ifonly1=True)
            Fmap2.append(fmap2.float())

            fmap2,fmap2u = self.fnet(image2)
            Fmap2.append(fmap2.float())

            fmap2 = self.fnet(image26,ifonly1=True)
            Fmap2.append(fmap2.float())

            fmap2 = self.fnet(image28,ifonly1=True)
            Fmap2.append(fmap2.float())

        #初始化光流和深度变化率
        coords0, coords1 = self.initialize_flow(image1)
        corr_fn8x = CorrpyBlockResScale(Fmap1, Fmap2, radius=4)
        exp = self.initialize_exp(image1).unsqueeze(1)
        x16coords0, x16coords1 = self.initialize_flow16x(image1)
        #初始化光流
        corr_fn16x = CorrBlock(fmap1u, fmap2u,num_levels=2, radius=6)
        now_time1 = time.time()
        with autocast(enabled=self.args.mixed_precision):
            # init flow，初始化部分
            corrf = corr_fn16x(x16coords1)
            net16x = self.init_flowconv(torch.cat([cnet16x, corrf], dim=1))


            flow_update = self.flow_head16x(net16x)
            # init exp
            exp_update = self.exp_head16x(net16x)

            #上采样只采样一次
            flow_update = F.upsample(flow_update, [flow_update.size()[2] * 2, flow_update.size()[3] * 2], mode='nearest')
            flow_update = self.flow_head8x(flow_update)

            exp_update = F.upsample(exp_update, [exp_update.size()[2] * 2, exp_update.size()[3] * 2],mode='nearest')
            exp_update = self.exp_head8x(exp_update)

            weight_update = .25 * self.upsample_weight(net)
            expweight_update = .25 * self.upsample_weightexp(net)
        now_time2 = time.time()
        #这里初始化光流和exp

        coords1 = coords1+flow_update
        flow_up = self.upsample_flow(flow_update, weight_update)
        flow_predictions.append(flow_up)

        exp = exp + 2*exp_update
        exp_up = self.upsample_exp(exp, expweight_update)
        exp_up = self.change_fun(exp_up)
        exp_predictions.append(exp_up)


        #开始迭代，次数可以搞得少一点，5次
        for itr in range(iters):
            coords1 = coords1.detach()
            exp = exp.detach()

            corr = corr_fn8x(coords1,exp)  # index correlation volume
            flow = coords1 - coords0
            now_time3 = time.time()
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask,scale_mask, delta_flow, dc_flow = self.update_block(net, context, corr, flow, exp,cnet16x)
            now_time4 = time.time()
            coords1 = coords1 + delta_flow
            exp = exp + 0.1*dc_flow
            # upsample predictions

            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            exp_up = self.upsample_exp(exp, scale_mask)
            exp_up = self.change_fun(exp_up)

            flow_predictions.append(flow_up)
            exp_predictions.append(exp_up)

        #最后一次非迭代更新
        exp = exp.detach()
        flow = (coords1 - coords0).detach()
        corr = corr_fn8x(coords1.detach(), exp)
        with autocast(enabled=self.args.mixed_precision):
            mask ,exp_update = self.update_blockUnet(net, context, corr, flow, exp,cnet16x)
        exp = exp + 0.05*exp_update
        exp_up = self.upsample_exp(exp, mask)
        exp_up = self.change_fun(exp_up)
        exp_predictions.append(exp_up)
        if test_mode:
            return coords1 - coords0, flow_up,exp_predictions[-1]
        return flow_predictions, exp_predictions
