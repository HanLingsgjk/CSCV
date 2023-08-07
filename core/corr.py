import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, coords_grid,bilinear_samplere

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass
import torch.nn as nn

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        #全精度的匹配
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        #匹配之后重采样
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)#这个是第一幅的坐标转移到第二幅图
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i].float()
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        batch, dim, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch, dim, ht1*wd1)
        fmap2 = fmap2.view(batch, dim, ht2*wd2)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2)
        return corr  / torch.sqrt(torch.tensor(dim).float())




class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

class CorrpyBlock4_3_343:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.corr_pyramid2 = []
        self.map1wh = []
        for dix in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]:
            # all pairs correlation
            corr = CorrBlock.corr(fmap1[int(dix[0])], fmap2[int(dix[1])])
            # F.upsample(dchange2, [im.size()[2], im.size()[3]], mode='bilinear')
            # 全精度的匹配
            batch, h1, w1, dim, h2, w2 = corr.shape
            self.map1wh.append([h1, w1])
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            # 匹配之后重采样
            self.corr_pyramid.append(corr)
        corr2 = self.corr_pyramid[2]
        for i in range(self.num_levels - 1):
            corr2 = F.avg_pool2d(corr2, 2, stride=2)
            self.corr_pyramid2.append(corr2)
        self.rate = [0.5, 0.75, 1, 1.25, 1.5]

    def __call__(self, coords, exp):
        r = 3
        r2 = 1
        r3 = 3
        coords = coords.permute(0, 2, 3, 1)  # 这个是第一幅的坐标转移到第二幅图
        exp = exp.squeeze(1)
        exp = exp.unsqueeze(3)  # 膨胀插帧
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        out_pyramid2 = []
        # 准备深度方向采样
        de = torch.linspace(-r2, r2, 2 * r2 + 1)
        de1 = torch.zeros(1)
        centrexp_lvl = exp.reshape(batch * h1 * w1, 1, 1, 1)
        delte = torch.stack(torch.meshgrid(de, de1), axis=-1).to(coords.device)
        delte_lvl = delte.view(1, 2 * r2 + 1, 1, 2)
        exp_lvl = delte_lvl + centrexp_lvl
        exp_lvl[:, :, :, 1] = 0

        # 尺度金字塔采样
        for i in [0, 1, 2, 3, 4]:
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r3, r3, 2 * r3 + 1) * self.rate[i]
            dy = torch.linspace(-r3, r3, 2 * r3 + 1) * self.rate[i]
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) * self.rate[i]

            delta_lvl = delta.view(1, 2 * r3 + 1, 2 * r3 + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            out_pyramid.append(corr)
        pyramid = torch.cat(out_pyramid, dim=1)
        pyramid = pyramid.view(batch * h1 * w1, 5, (2 * r3 + 1) * (2 * r3 + 1)).permute(0, 2, 1).unsqueeze(3)
        out = bilinear_samplere(pyramid, exp_lvl, mode='bilinear').view(batch, h1, w1,
                                                                        (2 * r3 + 1) * (2 * r3 + 1) * (2 * r2 + 1))

        '''
        import cv2
        pyramid6 = pyramid.view(batch,h1 , w1, (2 * r3 + 1)*(2 * r3 + 1),5)
        imgshow = pyramid6.detach().cpu().numpy()
        imshow24 = imgshow[0, :, :, 24, :]*0.071+imgshow[0, :, :, 23, :]*0.056+imgshow[0, :, :, 25, :]*0.056+imgshow[0, :, :, 16, :]*0.045+imgshow[0, :, :, 17, :]*0.056+imgshow[0, :, :, 18, :]*0.045+imgshow[0, :, :, 30, :]*0.045+imgshow[0, :, :, 31, :]*0.056+imgshow[0, :, :, 32, :]*0.045
        imgshow0 =imshow24[:,:,0]
        imgshow1 =imshow24[:,:,1]
        imgshow2 =imshow24[:,:,2]
        imgshow3 =imshow24[:,:,3]
        imgshow4 =imshow24[:,:,4]


        im0 = 255*(imgshow0-imshow24.min())/imshow24.max()
        im1 = 255*(imgshow1 - imshow24.min()) / imshow24.max()
        im2 = 255*(imgshow2 - imshow24.min()) / imshow24.max()
        im3 = 255*(imgshow3 - imshow24.min()) / imshow24.max()
        im4 = 255*(imgshow4 - imshow24.min()) / imshow24.max()
        cv2.imwrite('/home/xuxian/RAFT3D/pic_show/0.png', im0)
        cv2.imwrite('/home/xuxian/RAFT3D/pic_show/1.png', im1)
        cv2.imwrite('/home/xuxian/RAFT3D/pic_show/2.png', im2)
        cv2.imwrite('/home/xuxian/RAFT3D/pic_show/3.png', im3)
        cv2.imwrite('/home/xuxian/RAFT3D/pic_show/4.png', im4)

        k = 0.95
        plt.imshow(imgshow0, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        #plt.colorbar()
        plt.savefig('/home/xuxian/RAFT3D/pic_show/0.png')
        plt.show()

        plt.imshow(imgshow1, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        plt.savefig('/home/xuxian/RAFT3D/pic_show/1.png')
        plt.show()
        plt.imshow(imgshow2, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        plt.savefig('/home/xuxian/RAFT3D/pic_show/2.png')
        plt.show()
        plt.imshow(imgshow3, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        plt.savefig('/home/xuxian/RAFT3D/pic_show/3.png')
        plt.show()
        plt.imshow(imgshow4, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        plt.savefig('/home/xuxian/RAFT3D/pic_show/4.png')
        plt.show()
        '''

        # 原尺度的
        corr = self.corr_pyramid[2]
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        corr = corr.view(batch, h1, w1, -1)
        out_pyramid2.append(corr)
        for i in range(self.num_levels - 1):
            corr = self.corr_pyramid2[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** (i + 1)
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid2.append(corr)
        out2 = torch.cat(out_pyramid2, dim=-1)
        out3 = torch.cat([out2, out], dim=-1)

        return out3.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())