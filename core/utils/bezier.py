import numpy as np
from scipy.special import comb
import cv2
import  torch
import matplotlib.pyplot as plt
from core.utils.utils import bilinear_sampler, coords_grid

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i



def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array).astype(np.int32)
    yvals = np.dot(yPoints, polynomial_array).astype(np.int32)

    return xvals, yvals



if __name__ == "__main__":
    from matplotlib import pyplot as plt

    nPoints = np.random.randint(5,high=10)
    points = np.random.rand(nPoints,2)*200
    points[nPoints-1] = points[0]
    point_size = 1
    point_color = (1)  # BGR
    thickness = 4  # 0 、4、8
    aa = np.zeros([200,200], dtype=np.uint8)
    points = bezier_curve(points, nTimes=1000)
    pts = np.concatenate([points[0][:,np.newaxis],points[1][:,np.newaxis]],axis=1)

    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(aa,[pts],isClosed=True, color=(1), thickness=1)
    cv2.fillPoly(aa, [pts], color=(1))
    plt.imshow(aa)
    plt.show()
#size 是生成掩膜的大小，num是掩膜的数目
'''
掩膜生成步骤
1、利用bezier曲线生成掩膜xN   N,3,W,H  (x,y,d)(根据坐标采样)
2、将掩膜随机覆盖到原图像，初始化每个掩膜的(x,y,d)
3、生成随机的[R,T]并将其等效为透视变换矩阵，对于N个掩膜生成2N次
4、利用CV函数计算透视变换后的图像
5、汇聚随机掩膜，输出N,3,W,H  (x1,y1,d1)  (x2,y2,d2)
6、用掩膜替换掉原标注图的内容
'''
class mybezier():
    def __init__(self, size=[200,200],num = 3):
        super(mybezier, self).__init__()
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
    def get_mask(self,img1,last_im):
        H, W,_ = img1.shape
        scalerand = np.random.rand(1)*0.4+0.2
        scalerandf = np.random.rand(1)*100+100
        pointx = np.array([100,150,200,150,100,50,0,50])*scalerand
        pointy = np.array([200, 150, 100, 50 ,0, 50, 100, 150])*scalerand
        pointo = np.concatenate([pointx[: , np.newaxis],pointy[:,np.newaxis]],axis=1)
        aal = np.floor(200*scalerand).astype(np.uint8)[0]
        self.size[0] = aal
        self.size[1] = aal
        #生成随机形状掩膜
        nPoints = 8#np.random.randint(5, high=10)
        points = (np.random.rand(nPoints, 2)-0.5) * scalerandf * scalerand
        points = pointo + points
        #points[nPoints - 1] = points[0]
        aa = np.zeros([aal, aal], dtype=np.uint8)
        points = bezier_curve(points, nTimes=1000)
        pts = np.concatenate([points[0][:, np.newaxis], points[1][:, np.newaxis]], axis=1)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(aa, [pts], isClosed=True, color=(1), thickness=1)
        cv2.fillPoly(aa, [pts], color=(1))



        #初始化掩膜值（P1,P2,D）
        coords0 = coords_grid(1, self.size[0], self.size[1])
        #1、生成随机点
        x_p = np.random.randint(0, high=W-self.size[0]-1)
        y_p = np.random.randint(0, high=H-self.size[0]-1)
        coords0[:, 0, :, :] = coords0[:, 0, :, :] + x_p
        coords0[:, 1, :, :] = coords0[:, 1, :, :] + y_p

        #2、采样特征（r,g,b,d）
        img1n = (torch.from_numpy(last_im[:,:,:,np.newaxis])).permute(3,2, 0, 1)
        feature = img1n.float()

        d = torch.ones_like(coords0[:, 0:1, :, :])
        P = torch.cat([coords0, d], dim=1)
        coords1 = coords0.permute(0, 2, 3, 1).detach()
        coords1[:, :, :, 0] = coords1[:, :, :, 0] / (W / 2.) - 1
        coords1[:, :, :, 1] = coords1[:, :, :, 1] / (H / 2.) - 1
        feature2 = torch.nn.functional.grid_sample(feature, coords1,align_corners=True).view(3,self.size[0],self.size[1]).permute(1,2,0)
        feature2[aa==0,:]=0
        imgs = torch.zeros([H, W, 3])
        imgs[y_p:y_p+aal,x_p:x_p+aal,:] = feature2
        imgs=imgs.byte().numpy()

        #随机变换
        cx = float(W)/ 2
        cy = float(H)/ 2
        K0 = torch.zeros([3,3])
        K0[2,2] = 1
        K0[0,0] = 100
        K0[1,1] = 100
        K0[0,2] = cx
        K0[1,2] = cy
        K0 = K0

        t = torch.zeros([3,1])
        t[0] = torch.randn(1)*0.36
        t[1] = torch.randn(1)*0.36
        t[2]=  torch.randn(1)*0.1-0.07
        R = self.get_R()

        KD1 = torch.inverse(K0)
        P4 = torch.cat([P[:,:,0,0:1],P[:,:,-1,0:1],P[:,:,-1,-1:],P[:,:,0,-1:]],dim=2).view(3,4)
        x= torch.mm(KD1,P4)
        x = torch.mm(R,x)+t
        x = torch.mm(K0,x)/x[2,:]

        x2 = x
        #第二次转换
        t2 = torch.zeros([3, 1])
        t2[0] = torch.randn(1)*0.5
        t2[1] = torch.randn(1)*0.5
        t2[2] = (torch.randn(1)*0.1-0.07)*0.5
        R2 = self.get_R()
        x2 = torch.mm(KD1, x2)
        x2 = torch.mm(R2, x2) + t2
        x2 = torch.mm(K0, x2) / x2[2, :]


        former = P4.permute(1,0)[:,:2].numpy()
        pts = x.permute(1, 0)[:, :2].numpy()
        pts2 = x2.permute(1, 0)[:, :2].numpy()
        M = cv2.getPerspectiveTransform(former, pts)
        M2 = cv2.getPerspectiveTransform(pts, pts2)
        per1 = cv2.warpPerspective(imgs,M,(imgs.shape[1],imgs.shape[0]))#第一次转换后的图片
        per2 = cv2.warpPerspective(per1, M2, (imgs.shape[1], imgs.shape[0]))  # 第二次转换后的图片


        #计算深度变换和光流结果
        coords3 = coords_grid(1, H, W)
        dd = torch.ones_like(coords3[:,0:1,:,:])
        Pa = torch.cat([coords3,dd],dim=1)
        Pa = Pa.view(3,H*W)
        '''
        xa = torch.mm(KD1, Pa.clone())
        xa = torch.mm(R, xa) + t
        de1 = xa[2,:].view(H, W,1).clone()
        xa = torch.mm(K0, xa) / xa[2, :]
        Pe0 = xa[0:2,:].permute(1,0).view(H,W,2)
        '''
        Pe0 = Pa.clone()
        Pe0 = Pe0[0:2, :].permute(1, 0).view(H, W, 2)

        xb = torch.mm(KD1, Pa.clone())
        xb = torch.mm(R2, xb) + t2
        de2 = xb[2, :].view(H, W,1).clone()
        xb = torch.mm(K0, xb)[0:2, :] / xb[2, :]
        Pe1 = xb[0:2, :].permute(1,0).view(H, W,2)


        dc = (de2/1)
        flow = (Pe1-Pe0)
        ans = torch.cat([flow,dc],dim=2).numpy()
        ans[per1[:, :, 0]==0,:] = 0

        Pe0[per1[:, :, 0]==0,:] = 0
        Pe1[per1[:, :, 0] == 0,:] = 0
        pe00 = Pe0.numpy()
        pe01 = Pe1.numpy()

        if pe00[:,:,0].min()==0 and pe00[:,:,0].max()<W and pe00[:,:,1].min()==0 and pe00[:,:,1].max()<H:
            flag1 = 1
        else:
            flag1 = 0
        if pe01[:,:,0].min()==0 and pe01[:,:,0].max()<W and pe01[:,:,1].min()==0 and pe01[:,:,1].max()<H:
            flag2 = 1
        else:
            flag2 = 0

        return per1,per2,ans,flag1+flag2
    #这个掩膜用KITTI 3D的，考虑一下多视角合成可不可行？？？
    def get_mask_obj(self,img1,last_im,obj):
        H, W,_ = img1.shape
        Hm, Wm, _ = last_im.shape
        #首先确定这个模板图里面有啥
        maxnum = np.max(obj)
        uselist = []
        for i in range(1,maxnum+1):
            if np.sum(i==obj)!=0 and np.sum(i==obj)>500:
                uselist.append(i)
        if uselist.__len__()>=2:
            useidx = np.random.randint(0, uselist.__len__())

            things_num = uselist[useidx]
            self.coordsmask = coords_grid(1, Hm, Wm)
            mask_thing = things_num==obj
            idx_mask = mask_thing[np.newaxis,np.newaxis,:,:]*self.coordsmask.detach().numpy()
            idxxmax = int(np.max(idx_mask[0,0][mask_thing]))
            idxxmin = int(np.min(idx_mask[0,0][mask_thing]))
            idxymax = int(np.max(idx_mask[0,1][mask_thing]))
            idxymin = int(np.min(idx_mask[0,1][mask_thing]))
            aalx = idxxmax-idxxmin
            aaly = idxymax - idxymin

            self.size[0] = aalx
            self.size[1] = aaly
            if aaly < 200  and  aalx<800:
                masku = idx_mask[0, 0][idxymin:idxymax,idxxmin:idxxmax]

                #初始化掩膜值（P1,P2,D）
                coords0 = coords_grid(1, self.size[1], self.size[0])
                #1、确定采样位置
                coords0[:, 0, :, :] = coords0[:, 0, :, :] + idxxmin
                coords0[:, 1, :, :] = coords0[:, 1, :, :] + idxymin

                #2、采样特征（r,g,b,d）
                img1n = (torch.from_numpy(last_im[:,:,:,np.newaxis])).permute(3,2, 0, 1)
                feature = img1n.float()

                d = torch.ones_like(coords0[:, 0:1, :, :])
                P = torch.cat([coords0, d], dim=1)
                coords1 = coords0.permute(0, 2, 3, 1).detach()
                coords1[:, :, :, 0] = coords1[:, :, :, 0] / (Wm / 2.) - 1
                coords1[:, :, :, 1] = coords1[:, :, :, 1] / (Hm / 2.) - 1


                feature2 = torch.nn.functional.grid_sample(feature, coords1,align_corners=True)[0].permute(1,2,0)
                feature2[masku==0,:]=0
                #fs = feature2.detach().numpy().astype(np.uint8)
                imgs = torch.zeros([H, W, 3])
                #规划一个尽可能科学的随机偏移量
                py = 236-aaly
                px = 880-aalx
                ppx = np.random.randint(0,px)+20
                ppy = np.random.randint(0, py)+10
                imgs[ppy:(ppy+aaly),ppx:(ppx+aalx),:] = feature2
                imgs=imgs.byte().numpy()

                #随机变换
                cx = float(W)/ 2
                cy = float(H)/ 2
                K0 = torch.zeros([3,3])
                K0[2,2] = 1
                K0[0,0] = 100
                K0[1,1] = 100
                K0[0,2] = cx
                K0[1,2] = cy
                K0 = K0

                t = torch.zeros([3,1])
                t[0] = torch.randn(1)*0.3
                t[1] = torch.randn(1)*0.3
                t[2]=  0#torch.randn(1)*0.1-0.07
                R = self.get_R()

                KD1 = torch.inverse(K0)
                P4 = torch.cat([P[:,:,0,0:1],P[:,:,-1,0:1],P[:,:,-1,-1:],P[:,:,0,-1:]],dim=2).view(3,4)
                x= torch.mm(KD1,P4)
                x = torch.mm(R,x)+t
                x = torch.mm(K0,x)/x[2,:]

                x2 = x
                #第二次转换
                t2 = torch.zeros([3, 1])
                t2[0] = torch.randn(1)*0.5
                t2[1] = torch.randn(1)*0.5
                t2[2] = (torch.randn(1)*0.1-0.07)*0.5
                R2 = self.get_R()
                x2 = torch.mm(KD1, x2)
                x2 = torch.mm(R2, x2) + t2
                x2 = torch.mm(K0, x2) / x2[2, :]


                former = P4.permute(1,0)[:,:2].numpy()
                pts = x.permute(1, 0)[:, :2].numpy()
                pts2 = x2.permute(1, 0)[:, :2].numpy()
                M = cv2.getPerspectiveTransform(former, pts)
                M2 = cv2.getPerspectiveTransform(pts, pts2)
                per1 = cv2.warpPerspective(imgs,M,(imgs.shape[1],imgs.shape[0]))#第一次转换后的图片
                per2 = cv2.warpPerspective(per1, M2, (imgs.shape[1], imgs.shape[0]))  # 第二次转换后的图片

                #计算 深度变换和光流结果
                coords3 = coords_grid(1, H, W)
                dd = torch.ones_like(coords3[:,0:1,:,:])
                Pa = torch.cat([coords3,dd],dim=1)
                Pa = Pa.view(3,H*W)
                '''
                xa = torch.mm(KD1, Pa.clone())
                xa = torch.mm(R, xa) + t
                de1 = xa[2,:].view(H, W,1).clone()
                xa = torch.mm(K0, xa) / xa[2, :]
                Pe0 = xa[0:2,:].permute(1,0).view(H,W,2)
                '''
                Pe0 = Pa.clone()
                Pe0 = Pe0[0:2, :].permute(1, 0).view(H, W, 2)

                xb = torch.mm(KD1, Pa.clone())
                xb = torch.mm(R2, xb) + t2
                de2 = xb[2, :].view(H, W,1).clone()
                xb = torch.mm(K0, xb)[0:2, :] / xb[2, :]
                Pe1 = xb[0:2, :].permute(1,0).view(H, W,2)


                dc = (de2/1)
                #计算第二帧深度
                dcused = dc.detach().numpy()
                dcused[per1[:, :, 0] == 0, :] = 0
                dpre2 = cv2.warpPerspective(dcused, M2, (imgs.shape[1], imgs.shape[0]))


                flow = (Pe1-Pe0)
                ans = torch.cat([flow,dc],dim=2).numpy()
                ans[per1[:, :, 0]==0,:] = 0

                Pe0[per1[:, :, 0]==0,:] = 0
                Pe1[per1[:, :, 0] == 0,:] = 0
                pe00 = Pe0.numpy()
                pe01 = Pe1.numpy()
                if pe00[:,:,0].min()==0 and pe00[:,:,0].max()<W and pe00[:,:,1].min()==0 and pe00[:,:,1].max()<H:
                    flag1 = 1
                else:
                    flag1 = 0
                if (pe01[:,:,0].max()>=0 and pe01[:,:,1].max()>=0) or (pe01[:,:,0].min()<W  and pe01[:,:,1].min()<H):
                    flag2 = 1
                else:
                    flag2 = 0

                return per1,per2,ans,dpre2,flag1+flag2
            return 0, 0, 0, 0, 1
        else:
            return 0, 0, 0, 0, 1


