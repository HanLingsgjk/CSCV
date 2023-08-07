import numpy as np
from scipy.special import comb
import cv2
import  torch
import matplotlib.pyplot as plt
from core.utils.utils import bilinear_sampler, coords_grid

'''
掩膜生成步骤
1、生成矩形掩膜
2、使用随机数填充掩膜，初始化每个掩膜的(x,y,d)，深度随机
3、生成随机的[R,T]并将其等效为透视变换矩阵，对于N个掩膜生成2N次
4、利用CV函数计算透视变换后的图像
5、汇聚随机掩膜，输出N,3,W,H  (x1,y1,d1)  (x2,y2,d2)
6、用掩膜替换掉原标注图的内容
'''
class retangle():
    def __init__(self, size=[100,100],num = 3):
        super(retangle, self).__init__()
        self.size = size
        self.num = num
    def get_R(self):
        Rx = torch.eye(3)
        Ry = torch.eye(3)
        Rz = torch.eye(3)
        angle = torch.randn(3)*0.05


        Rx[1, 1] = torch.cos(angle[0])
        Rx[1, 2] = torch.sin(angle[0])
        Rx[2, 1] = -torch.sin(angle[0])
        Rx[2, 2] = torch.cos(angle[0])

        Ry[0, 0] = torch.cos(angle[1])
        Ry[0, 2] = -torch.sin(angle[1])
        Ry[2, 0] = torch.sin(angle[1])
        Ry[2, 2] = torch.cos(angle[1])

        Rz[0, 0] = torch.cos(angle[2])
        Rz[0, 1] = torch.sin(angle[2])
        Rz[1, 0] = -torch.sin(angle[2])
        Rz[1, 1] = torch.cos(angle[2])

        R = torch.mm(torch.mm(Rx,Ry),Rz)
        return R
    def get_mask(self,img1):
        H, W,_ = img1.shape

        mask_noise = np.random.randint(0, 255, [self.size[0], self.size[1], 3])
        coords0 = coords_grid(1, self.size[0], self.size[1])
        x_p = np.random.randint(self.size[0], high=W - self.size[0] - 1)
        y_p = np.random.randint(self.size[1], high=H - self.size[1] - 1)
        coords0[:, 0, :, :] = coords0[:, 0, :, :] + x_p
        coords0[:, 1, :, :] = coords0[:, 1, :, :] + y_p
        d = torch.ones_like(coords0[:, 0:1, :, :])
        P = torch.cat([coords0, d], dim=1)
        imgs = torch.zeros([H, W, 3])
        imgs = imgs.byte().numpy()
        imgs[y_p:y_p + self.size[0], x_p:x_p + self.size[1], :] = mask_noise

        # 随机变换
        cx = float(W) / 2
        cy = float(H) / 2
        K0 = torch.zeros([3, 3])
        K0[2, 2] = 1
        K0[0, 0] = 100
        K0[1, 1] = 100
        K0[0, 2] = cx
        K0[1, 2] = cy
        K0 = K0
        KD1 = torch.inverse(K0)
        P4 = torch.cat([P[:, :, 0, 0:1], P[:, :, -1, 0:1], P[:, :, -1, -1:], P[:, :, 0, -1:]], dim=2).view(3, 4)

        x = P4
        x2 = x
        # 第二次转换
        t2 = torch.zeros([3, 1])
        t2[0] = torch.randn(1) * 0.5
        t2[1] = torch.randn(1) * 0.1
        t2[2] = (torch.randn(1) * 0.5).clamp(-0.5,0.5)
        R2 = self.get_R()
        x2 = torch.mm(KD1, x2)
        x2 = x2 + t2#torch.mm(R2, x2)
        x2 = torch.mm(K0, x2)
        x2 = x2/ x2[2, :]
        former = P4.permute(1, 0)[:, :2].numpy()
        pts = x.permute(1, 0)[:, :2].numpy()
        pts2 = x2.permute(1, 0)[:, :2].numpy()
        M = cv2.getPerspectiveTransform(former, pts)
        M2 = cv2.getPerspectiveTransform(pts, pts2)
        per1 = cv2.warpPerspective(imgs, M, (imgs.shape[1], imgs.shape[0]))  # 第一次转换后的图片
        per2 = cv2.warpPerspective(per1, M2, (imgs.shape[1], imgs.shape[0]))  # 第二次转换后的图片

        # 计算深度变换和光流结果
        coords3 = coords_grid(1, H, W)
        dd = torch.ones_like(coords3[:, 0:1, :, :])
        Pa = torch.cat([coords3, dd], dim=1)
        Pa = Pa.view(3, H * W)

        Pe0 = Pa.clone()
        Pe0 = Pe0[0:2, :].permute(1, 0).view(H, W, 2)

        xb = torch.mm(KD1, Pa.clone())
        xb = xb + t2#torch.mm(R2, xb) + t2
        de2 = xb[2, :].view(H, W, 1).clone()
        xb = torch.mm(K0, xb)[0:2, :] / xb[2, :]
        Pe1 = xb[0:2, :].permute(1, 0).view(H, W, 2)

        dc = (de2 / 1)
        flow = (Pe1 - Pe0)
        ans = torch.cat([flow, dc], dim=2).numpy()
        ans[per1[:, :, 0] == 0, :] = 0

        Pe0[per1[:, :, 0] == 0, :] = 0
        Pe1[per1[:, :, 0] == 0, :] = 0
        pe00 = Pe0.numpy()
        pe01 = Pe1.numpy()
        if pe00[:, :, 0].min() == 0 and pe00[:, :, 0].max() < W and pe00[:, :, 1].min() == 0 and pe00[:, :,
                                                                                                 1].max() < H:
            flag1 = 1
        else:
            flag1 = 0
        if pe01[:, :, 0].min() == 0 and pe01[:, :, 0].max() < W and pe01[:, :, 1].min() == 0 and pe01[:, :,
                                                                                                 1].max() < H:
            flag2 = 1
        else:
            flag2 = 0

        return per1, per2, ans, flag1 + flag2

