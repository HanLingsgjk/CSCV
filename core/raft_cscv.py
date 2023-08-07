import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.update import BasicUpdateBlock, SmallUpdateBlock, ScaleflowUpdateBlock, DCUpdateBlock
from core.extractor import BasicEncoder, SmallEncoder
from core.corr import CorrpyBlock4_3_343, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from core.utils.resnet import FPN

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


class RAFT343used(nn.Module):
    def __init__(self, args):
        super(RAFT343used, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 192
            self.context_dim = cdim = 192
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = ScaleflowUpdateBlock(self.args, hidden_dim=hdim)
        self.dc_block = DCUpdateBlock(self.args,hdim)
        # blur1 = cv2.resize(blur1, (int(w * (xita / 2)), int(h * (xita / 2))))
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

        #归一化层   N, C, H, W = 20, 5, 10, 10

        self.layer_normconv11 = nn.Conv2d(256, 256, kernel_size=1)
        self.layer_normconv23 = nn.Conv2d(256, 256, kernel_size=1)

        self.layer_normconv21 = nn.Conv2d(256, 256, kernel_size=1)
        self.layer_normconv22 = nn.Conv2d(256, 256, kernel_size=1)
        self.layer_normconv24 = nn.Conv2d(256, 256, kernel_size=1)
        self.layer_normconv25 = nn.Conv2d(256, 256, kernel_size=1)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        cs = coords0.detach().cpu().numpy()
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
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
    def upsample9_exp(self, exp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = exp.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_exp = exp.view(N, 1, 9, 1, 1, H, W)

        up_exp = torch.sum(mask * up_exp, dim=2)
        up_exp = up_exp.permute(0, 1, 4, 2, 5, 3)
        return up_exp.reshape(N, 1, 8 * H, 8 * W)
    def change_fun(self, exp):  # 将坐标转换成膨胀率
        exp = exp.clamp(0, 4)
        x = exp * 0.25 + 0.5
        return x
    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        _, _, h, w = image1.shape
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        w1 = self.weightt4
        image21 = F.conv2d(image2, w1, padding=2, groups=3)
        image21 = F.interpolate(image21, [int(h * self.delta4), int(w * self.delta4)])
        w2 = self.weightt2
        image23 = F.conv2d(image2, w2, padding=2, groups=3)
        image23 = F.interpolate(image23, [int(h * self.delta2), int(w * self.delta2)])

        image26 = F.interpolate(image2, [int(h * 1.25), int(w * 1.25)])
        image28 = F.interpolate(image2, [int(h * 1.5), int(w * 1.5)])

        # 构建连续尺度的高斯金字塔
        # 1、高斯模糊
        # 2、降采样

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        Fmap1 = []
        Fmap2 = []
        layer_norm = False
        layer_norm_affine = False
        # run the feature network
        now_time1 = time.time()
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet(image1)
            layer_n = nn.LayerNorm(fmap1.shape[-3:],elementwise_affine=False).cuda()
            if layer_norm:
                fmap1 = layer_n(fmap1.float())
            if layer_norm_affine:
                fmap1 = self.layer_normconv11(fmap1)
            Fmap1.append(fmap1.float())

            #fmap2 = self.fnet(image21)
            fmap2 = self.fnet(image21)
            layer_n = nn.LayerNorm(fmap2.shape[-3:],elementwise_affine=False).cuda()
            if layer_norm:
                fmap2 = layer_n(fmap2.float())
            if layer_norm_affine:
                fmap2 = self.layer_normconv21(fmap2)
            Fmap2.append(fmap2.float())

            #fmap2 = self.fnet(image23)
            fmap2 = self.fnet(image23)
            layer_n = nn.LayerNorm(fmap2.shape[-3:],elementwise_affine=False).cuda()
            if layer_norm:
                fmap2 = layer_n(fmap2.float())
            if layer_norm_affine:
                fmap2 = self.layer_normconv22(fmap2)
            Fmap2.append(fmap2.float())

            fmap2 = self.fnet(image2)

            layer_n = nn.LayerNorm(fmap2.shape[-3:],elementwise_affine=False).cuda()
            if layer_norm:
                fmap2 = layer_n(fmap2.float())
            if layer_norm_affine:
                fmap2 = self.layer_normconv23(fmap2)
            Fmap2.append(fmap2.float())

            #fmap2 = self.fnet(image26)
            fmap2 = self.fnet(image26)
            layer_n = nn.LayerNorm(fmap2.shape[-3:],elementwise_affine=False).cuda()
            if layer_norm:
                fmap2 = layer_n(fmap2.float())
            if layer_norm_affine:
                fmap2 = self.layer_normconv24(fmap2)
            Fmap2.append(fmap2.float())

            #fmap2 = self.fnet(image28)
            fmap2 = self.fnet(image28)

            layer_n = nn.LayerNorm(fmap2.shape[-3:],elementwise_affine=False).cuda()
            if layer_norm:
                fmap2 = layer_n(fmap2.float())
            if layer_norm_affine:
                fmap2 = self.layer_normconv25(fmap2)
            Fmap2.append(fmap2.float())

        now_time2 = time.time()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrpyBlock4_3_343(Fmap1, Fmap2, radius=self.args.corr_radius)

        now_time3 = time.time()
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        now_time4 = time.time()

        coords0, coords1 = self.initialize_flow(image1)
        exp = self.initialize_exp(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        exp = exp.unsqueeze(1)
        flow_predictions = []
        exp_predictions = []

        for itr in range(iters):
            coords1 = coords1.detach()
            exp = exp.detach()

            corr = corr_fn(coords1,exp)  # index correlation volume
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                net, up_mask,scale_mask, delta_flow, dc_flow = self.update_block(net, inp, corr, flow, exp)

            coords1 = coords1 + delta_flow
            exp = exp + dc_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)

            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                exp_up = self.upsample_exp(exp, scale_mask)
                exp_up = self.change_fun(exp_up)
            flow_predictions.append(flow_up)
            exp_predictions.append(exp_up)

        now_time5 = time.time()

        # 最后修正一下dc
        exp = exp.detach()
        flow = (coords1 - coords0).detach()
        corr = corr_fn(coords1.detach(),exp)

        up_maskdc, dc_flowe = self.dc_block(net, inp, corr, flow, exp)
        exp = exp + dc_flowe * 0.005
        exp_up = self.upsample_exp(exp, up_maskdc)
        exp_up = self.change_fun(exp_up)
        exp_predictions.append(exp_up)

        now_time6 = time.time()
        '''
        ims1 = (128*(image1[0]+1)).permute(1,2,0).detach().cpu().numpy().astype(int)
        plt.imshow(ims1)
        plt.show()
        ims2 = (128*(image2[0]+1)).permute(1,2,0).detach().cpu().numpy().astype(int)
        plt.imshow(ims2)
        plt.show()
        fshow = flow_up.detach().cpu().numpy()
        plt.imshow(fshow[0,0])
        plt.show()
        plt.imshow(fshow[0,1])
        plt.show()
        expshow = exp_up.detach().cpu().numpy()
        plt.imshow(expshow[0, 0])
        plt.show()
        '''
        print(now_time2-now_time1,now_time3-now_time2,now_time4-now_time3,now_time5-now_time4,now_time6-now_time5)
        if test_mode:
            return coords1 - coords0, flow_up, exp_up

        return flow_predictions, exp_predictions