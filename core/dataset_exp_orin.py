# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
#原汁原味的DATASETS
from utils.utils import  coords_grid
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import os
import pickle
import math
import random
from glob import glob
import os.path as osp
import matplotlib.pyplot as plt
from core.utils.rectangle_noise import retangle
from core.utils import frame_utils
import  cv2
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentorm
def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth
def readPFM(file):
    import re
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:
        return readPFM(path)[0]
def get_grid_np(B,H,W):
    meshgrid_base = np.meshgrid(range(0, W), range(0, H))[::-1]
    basey = np.reshape(meshgrid_base[0], [1, 1, 1, H, W])
    basex = np.reshape(meshgrid_base[1], [1, 1, 1, H, W])
    grid = torch.tensor(np.concatenate((basex.reshape((-1, H, W, 1)), basey.reshape((-1, H, W, 1))), -1)).float()
    return grid.view( H, W, 2)
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorm(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.driving = False
        self.is_test = False
        self.init_seed = False
        self.test_scene = False
        self.stereo = False
        self.flow_list = []
        self.dispnet =[]
        self.depth_list = []
        self.image_list = []
        self.extra_info = []
        self.mask_list = []
        self.occ_list = []
        self.rect = retangle()
        self.kit = 0
        self.k = 1
        self.kr = 0
        self.get_depth = 0
        self.kitti_test = 0
        self.sintel_test = 0

        self.last_image = np.random.randn(320,960,3)
    def __getitem__(self, index):
        self.kit = self.kit +1
        if self.test_scene:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            dispnet = np.abs(disparity_loader(self.dispnet[index]))
            return img1, img2, self.extra_info[index],dispnet
        if self.is_test and not self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        if self.get_depth:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            mask[dc_change > 1.5] = 0
            mask[dc_change < 0.5] = 0
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
            #读取光流结果
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            for i in range(int(self.kr)):
                imgb1, imgb2, ansb, flag = self.rect.get_mask(img1)
                if flag > 1:
                    img1[imgb1 > 0] = imgb1[imgb1 > 0]
                    img2[imgb2 > 0] = imgb2[imgb2 > 0]
                    flow[imgb1[:, :, 0] > 0, :] = ansb[imgb1[:, :, 0] > 0, :2]
                    dc_change[imgb1[:, :, 0] > 0, 0:1] = ansb[imgb1[:, :, 0] > 0, 2:]
                    d1[imgb1[:, :, 0] > 0] = 10
                    d2[imgb1[:, :, 0] > 0] = dc_change[imgb1[:, :, 0] > 0,0]*10
                    li = ansb[:, :, 2] > 0
                    dc_change[li, 1] = 1
                    mask[imgb1[:, :, 0] > 0]=2

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0

            return img1,img2,flow,dc_change,d1,d2,disp1,disp2,mask,self.extra_info[index]#这个mask是是否有噪音块的掩膜
        if self.kitti_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            #mask = frame_utils.read_gen(self.mask_list[index])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            # 读取光流结果
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0
            return img1, img2, flow, dc_change, d1, d2, disp1, disp2, mask,valid, self.extra_info[index]  # 这个mask是是否有噪音块的掩膜
        if self.sintel_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            #mask = frame_utils.read_gen(self.mask_list[index])
            d1, d2, mask = self.get_dc(index)
            dc_change = d2 / d1
            d1[mask == 0] = 0
            d2[mask == 0] = 0
            dc_change[mask == 0] = 0
            # 读取光流结果
            flow = frame_utils.read_gen(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            mask = np.array(mask).astype(np.uint8)
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            disp1 = self.depth_to_disp(d1)
            disp2 = self.depth_to_disp(d2)
            disp1[mask == 0] = 0
            disp2[mask == 0] = 0
            return img1, img2, flow, dc_change, d1, d2, disp1, disp2, mask,0, self.extra_info[index]
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        d1,d2,mask = self.get_dc(index)

        dc_change = d2/d1
        mask[dc_change>1.5] = 0
        mask[dc_change <0.5] = 0
        dc_change[mask==0]=0
        if self.occlusion:
            dcc = dc_change
            dcc = abs(cv2.filter2D(dcc,-1,kernel=self.kernel2))
            maskd = torch.from_numpy(dcc>1).bool()
            dc_change[maskd!=0] = 0
            masku = dc_change>0
            #再加一个遮挡
            dc_change = np.concatenate((dc_change[:,:,np.newaxis],masku[:,:,np.newaxis]),axis =2 )
        else:
            dc_change = np.concatenate((dc_change[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
        if self.sparse:
            if self.driving:
                flow, valid = frame_utils.readFlowdriving(self.flow_list[index])
            elif self.stereo:
                flowx = disparity_loader(self.depth_list[index][0])
                flow = np.concatenate((flowx[:, :, np.newaxis], flowx[:, :, np.newaxis]), axis=2)
                valid = flowx>0
                flow[:,:,1]=0
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])

        else:
            flow = frame_utils.read_gen(self.flow_list[index])


        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        dc_change = np.array(dc_change).astype(np.float32)


        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]


        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow,dc_change, valid = self.augmentor(img1, img2, flow,dc_change, valid)
            else:
                img1, img2, flow, dc_change = self.augmentor(img1, img2, flow,dc_change)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        dc_change   = torch.from_numpy(dc_change).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, dc_change, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.depth_list = v * self.depth_list
        self.occ_list = v * self.occ_list
        return self

    def __len__(self):
        return len(self.image_list)

class MpiSinteltest(FlowDataset):#/home/lh/RAFT-master/dataset/Sintel
    def __init__(self, aug_params=None, split='training', root='/new_data/kitti_data/datasets/sintel', dstype='clean'):
        super(MpiSinteltest, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')
        occ_root = osp.join(root, split, 'occlusions')
        self.occlusion = True
        self.sintel_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
            occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                if split != 'test':
                    self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                if self.occlusion:
                    for i in range(len(image_list) - 1):
                        self.occ_list += [occ_list[i]]
    def get_dc(self,index):
        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()

        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        h, w, c = flow.shape

        depth1 = torch.tensor(depth_read(self.depth_list[index][0]))
        depth2 = torch.tensor(depth_read(self.depth_list[index][1])).view(1, 1, h, w)
        flowg = torch.tensor(flow)
        frep = get_grid_np(c, h, w)
        frepb = (frep + flowg).view(1, h, w, 2)
        frepb[:, :, :, 0] = frepb[:, :, :, 0] / (w / 2.) - 1
        frepb[:, :, :, 1] = frepb[:, :, :, 1] / (h / 2.) - 1
        depth2 = (torch.nn.functional.grid_sample(depth2, frepb,mode='nearest').view(h, w))
        depth2 = depth2.view(h, w)
        return depth1.numpy(),depth2.numpy(),1-occ.numpy()
    def depth_to_disp(self,Z, bl=1, fl=1000):
        disp = bl * fl / Z
        return disp

class MpiSintel(FlowDataset):#/home/lh/RAFT-master/dataset/Sintel
    def __init__(self, aug_params=None, split='training', root='/home/xuxian/RAFT3D/datasets/sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        depth_root = osp.join(root, split, 'depth')
        occ_root = osp.join(root, split, 'occlusions')
        self.occlusion = True
        if split == 'test':
            self.is_test = True
        self.kernel = np.ones([5, 5], np.float32)
        self.kernel2 = np.ones([3, 3], np.float32)*-1
        self.kernel2[1,1] = 8
        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            depth_list = sorted(glob(osp.join(depth_root, scene, '*.dpt')))
            occ_list = sorted(glob(osp.join(occ_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                if split != 'test':
                    self.depth_list += [[depth_list[i], depth_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                if self.occlusion:
                    for i in range(len(image_list) - 1):
                        self.occ_list += [occ_list[i]]
    def get_dc(self,index):
        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()
            #膨胀occ
            '''
            acc = occ.numpy().astype(np.uint8)
            occ = cv2.filter2D(acc,-1,kernel=self.kernel)
            occ = torch.from_numpy(occ>0).bool()
            '''
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        h, w, c = flow.shape

        depth1 = torch.tensor(depth_read(self.depth_list[index][0]))
        depth2 = torch.tensor(depth_read(self.depth_list[index][1])).view(1, 1, h, w)
        flowg = torch.tensor(flow)
        frep = get_grid_np(c, h, w)
        frepb = (frep + flowg).view(1, h, w, 2)
        frepb[:, :, :, 0] = frepb[:, :, :, 0] / (w / 2.) - 1
        frepb[:, :, :, 1] = frepb[:, :, :, 1] / (h / 2.) - 1
        depth2 = (torch.nn.functional.grid_sample(depth2, frepb,mode='nearest').view(h, w))
        depth2 = depth2.view(h, w)
        return depth1.numpy(),depth2.numpy(),1-occ.numpy()
class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/home/lh/RAFT-master/dataset/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]
class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/new_data/flyingtings/flyingthings/flyingthings/', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)
        exclude = np.loadtxt('/home/xuxian/RAFT-3D-master/RAFT-3D-master/misc/exclude.txt', delimiter=' ', dtype=np.unicode_)
        exclude = set(exclude)
        self.occlusion = False
        self.driving = True
        for cam in ['left','right']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                d0_dirs = sorted(glob(osp.join(root, 'disparity/TRAIN/*/*')))
                d0_dirs = sorted([osp.join(f, cam) for f in d0_dirs])

                dc_dirs = sorted(glob(osp.join(root, 'disparity_change/TRAIN/*/*')))
                dc_dirs = sorted([osp.join(f, direction, cam) for f in dc_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir,d0dir,dcdir in zip(image_dirs, flow_dirs,d0_dirs,dc_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    d0s = sorted(glob(osp.join(d0dir, '*.pfm')))
                    dcs = sorted(glob(osp.join(dcdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        tag = '/'.join(images[i].split('/')[-5:])
                        if tag in exclude:
                            print("Excluding %s" % tag)
                            continue
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                            self.depth_list += [[d0s[i], dcs[i]]]
                            frame_id = images[i].split('/')[-1]
                            self.extra_info += [[frame_id]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]
                            self.depth_list += [[d0s[i+1], dcs[i+1]]]
                            frame_id = images[i+1].split('/')[-1]
                            self.extra_info += [[frame_id]]
    def triangulation(self, disp, bl=1):#kitti flow 2015

        fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1])+d1)
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask

class KITTI(FlowDataset):#/home/lh/RAFT_master/dataset/kitti_scene   '/home/xuxian/RAFT3D/datasets'
    def __init__(self, aug_params=None, split='training', root='/new_data/kitti_data/datasets',get_depth=0):
        super(KITTI, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        if split == 'testing':
            self.is_test = True
        if split == 'submit':
            self.is_test = True
        if split == 'submitother':
            self.is_test = True
        if split =='test':
            self.test_scene = True
        self.occlusion = False
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]
        if split == 'training':
            root = osp.join(root, split)
            images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

            for j in range(images2o.__len__()):
                #if j%5>0 or self.get_depth:
                    images1.append(images1o[j])
                    images2.append(images2o[j])
                    disp1.append(disp1o[j])
                    disp2.append(disp2o[j])
        elif split=='testing':
            root = osp.join(root, 'training')
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        elif split=='submit':
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        elif split=='test':
            images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
            disp1LEA = sorted(glob(osp.join(root, 'disp_ganet_testing/*_10.png')))
            self.dispnet = disp1LEA
        else:
            images1 = sorted(glob(osp.join(root, '*.jpg')))
            images2 = images1[1:]
            images1.pop()
            disp1 = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            disp2 = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]
        if split == 'training':
            flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            for j in range(flowo.__len__()):
                #if j%5>0 or self.get_depth:
                    flow.append(flowo[j])
        elif split == 'testing':
            flow = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        self.flow_list = flow


    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask
class KITTI_test200(FlowDataset):#/home/lh/RAFT3D-DEPTH/data_train_test /home/xuxian/RAFT3D/data_train_test
    def  __init__(self, aug_params=None, split='kitti_test', root='/new_data/kitti_data/datasets/training',get_depth=0):
        super(KITTI_test200, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        if split == 'kitti_test':
           self.kitti_test = 1
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]
        mask = []
        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        masko = sorted(glob(osp.join(root, 'mask_img/*_10.png')))
        for j in range(images2o.__len__()):
            images1.append(images1o[j])
            images2.append(images2o[j])
            disp1.append(disp1o[j])
            disp2.append(disp2o[j])
            mask.append(masko[j])


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]

        flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        for j in range(flowo.__len__()):
                flow.append(flowo[j])

        self.flow_list = flow
        self.mask_list = mask

    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask
class KITTI_test(FlowDataset):#/home/lh/RAFT3D-DEPTH/data_train_test /home/xuxian/RAFT3D/data_train_test#/new_data/kitti_data/datasets/training
    def  __init__(self, aug_params=None, split='kitti_test', root='/new_data/kitti_data/datasets/training',get_depth=0):
        super(KITTI_test, self).__init__(aug_params, sparse=True)
        self.get_depth=get_depth
        self.occlusion = False
        if split == 'kitti_test':
           self.kitti_test = 1
        images1 =[]
        images2 =[]
        disp1 = []
        disp2 = []
        flow =[]
        mask = []
        #images1o = sorted(glob(osp.join(root, 'img1/*_10.png')))
        #images2o = sorted(glob(osp.join(root, 'img2/*_10.png')))
        #disp1o = sorted(glob(osp.join(root, 'disp_0/*_10.png')))
        #disp2o = sorted(glob(osp.join(root, 'disp_1/*_10.png')))
        #masko = sorted(glob(osp.join(root, 'mask/*_10.png')))
        images1o = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2o = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        disp1o = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        disp2o = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))
        masko = sorted(glob(osp.join(root, 'mask_img/*_10.png')))
        for j in range(images2o.__len__()):
            images1.append(images1o[j])
            images2.append(images2o[j])
            disp1.append(disp1o[j])
            disp2.append(disp2o[j])
            mask.append(masko[j])


        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        for disps1, disps2 in zip(disp1, disp2):
            self.depth_list += [[disps1, disps2]]

        #flowo = sorted(glob(osp.join(root, 'flow/*_10.png')))
        flowo = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        for j in range(flowo.__len__()):
                flow.append(flowo[j])

        self.flow_list = flow
        self.mask_list = mask

    def triangulation(self, disp, bl=0.5327254279298227, fl=721.5377):#kitti flow 2015
        disp[disp==0]= 1
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z
    def depth_to_disp(self,Z, bl=0.5327254279298227, fl=721.5377):
        disp = bl * fl / Z
        return disp

    #获取有效区域的掩膜，以及两个深度
    def get_dc(self,index):

        d1 = disparity_loader(self.depth_list[index][0])
        d2 = disparity_loader(self.depth_list[index][1])
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1),self.triangulation(d2),mask
class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='/home/lh/RAFT-master/dataset/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1
class Driving(FlowDataset):
    def __init__(self, aug_params=None,  split='training',root='/new_data/driving'):
        super(Driving, self).__init__(aug_params, sparse=True)
        self.calib = []
        self.occlusion = False
        self.driving = True
        level_stars = '/*' * 6
        candidate_pool = glob('%s/optical_flow%s' % (root, level_stars))
        for flow_path in sorted(candidate_pool):
            idd = flow_path.split('/')[-1].split('_')[-2]
            if 'into_future' in flow_path:
                idd_p1 = '%04d' % (int(idd) + 1)
            else:
                idd_p1 = '%04d' % (int(idd) - 1)
            if os.path.exists(flow_path.replace(idd, idd_p1)):
                d0_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','disparity')
                d0_path = '%s/%s.pfm' % (d0_path.rsplit('/', 1)[0], idd)
                dc_path = flow_path.replace('optical_flow', 'disparity_change')
                dc_path = '%s/%s.pfm' % (dc_path.rsplit('/', 1)[0], idd)
                im_path = flow_path.replace('/into_future/', '/').replace('/into_past/', '/').replace('optical_flow','frames_cleanpass')
                im0_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd)
                im1_path = '%s/%s.png' % (im_path.rsplit('/', 1)[0], idd_p1)
                frame_id = im1_path.split('/')[-1]
                self.extra_info += [[frame_id]]
                #calib.append('%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0]))
                self.flow_list += [flow_path]
                self.image_list += [[im0_path,im1_path]]
                self.depth_list += [[d0_path,dc_path]]
                self.calib +=['%s/camera_data.txt' % (im0_path.replace('frames_cleanpass', 'camera_data').rsplit('/', 2)[0])]
    def triangulation(self, disp,index, bl=1):#kitti flow 2015
        if '15mm_' in self.calib[index]:
            fl = 450  # 450
        else:
            fl = 1050
        depth = bl * fl / disp  # 450px->15mm focal length
        Z = depth
        return Z

    def get_dc(self,index):
        d1 = np.abs(disparity_loader(self.depth_list[index][0]))
        d2 = np.abs(disparity_loader(self.depth_list[index][1])+d1)
        flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(flow[:, :, 0] != 0, flow[:, :, 1] != 0), d1 != 0), d2 != 0).astype(float)

        return self.triangulation(d1,index),self.triangulation(d2,index),mask
def fetch_dataloader(args, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        driving = Driving(aug_params, split='training')
        train_dataset = clean_dataset+driving

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti + 5 * hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100 * sintel_clean + 100 * sintel_final+things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        kitti = KITTI(aug_params, split='training')
        train_dataset = 200*kitti

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=6, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = \
        intrinsics[None].unbind(dim=-1)

    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(),
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[..., 0]
    y1 = y0 + flow2d[..., 1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = torch.stack([X1 - X0, Y1 - Y0, Z1 - Z0], dim=-1)
    return flow3d


#这个是用于在RAFT3D上面测试的
class FlyingThingsTest(data.Dataset):
    def __init__(self, root='/new_data/flyingtings/flyingthings/flyingthings/'):

        self.dataset_index = []
        test_data = pickle.load(open('/new_data/flyingtings/flyingthings/flyingthings/things_test_data.pickle', 'rb'))

        for (data_paths, sampled_pix1_x, sampled_pix2_y, mask) in test_data:
            split, subset, sequence, camera, frame = data_paths.split('_')
            sampled_pix1_x = sampled_pix1_x[mask]
            sampled_pix2_y = 539 - sampled_pix2_y[mask]
            sampled_index = np.stack([sampled_pix2_y, sampled_pix1_x], axis=0)

            # intrinsics
            fx, fy, cx, cy = (1050.0, 1050.0, 480.0, 270.0)
            intrinsics = np.array([fx, fy, cx, cy])

            frame = int(frame)
            image1 = osp.join(root, 'frames_cleanpass', split, subset, sequence, camera, "%04d.png" % (frame))
            image2 = osp.join(root, 'frames_cleanpass', split, subset, sequence, camera, "%04d.png" % (frame + 1))

            disp1 = osp.join(root, 'disparity', split, subset, sequence, camera, "%04d.pfm" % (frame))
            disp2 = osp.join(root, 'disparity', split, subset, sequence, camera, "%04d.pfm" % (frame + 1))

            if camera == 'left':
                flow = osp.join(root, 'optical_flow', split, subset, sequence,
                                'into_future', camera, "OpticalFlowIntoFuture_%04d_L.pfm" % (frame))
                disparity_change = osp.join(root, 'disparity_change', split, subset,
                                            sequence, 'into_future', camera, "%04d.pfm" % (frame))

            else:
                flow = osp.join(root, 'optical_flow', split, subset, sequence,
                                'into_future', camera, "OpticalFlowIntoFuture_%04d_R.pfm" % (frame))
                disparity_change = osp.join(root, 'disparity_change', split, subset,
                                            sequence, 'into_future', camera, "%04d.pfm" % (frame))

            datum = (image1, image2, disp1, disp2, flow, disparity_change, intrinsics, sampled_index)
            self.dataset_index.append(datum)

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):

        image1, image2, disp1, disp2, flow, disparity_change, intrinsics, sampled_index = self.dataset_index[index]

        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        disp1 = frame_utils.read_gen(disp1)
        disp2 = frame_utils.read_gen(disp2)

        flow2d = frame_utils.read_gen(flow)[..., :2]
        disparity_change = frame_utils.read_gen(disparity_change)

        depth1 = torch.from_numpy(intrinsics[0] / disp1).float()
        depth2 = torch.from_numpy(intrinsics[0] / disp2).float()

        # transformed depth
        depth12 = torch.from_numpy(intrinsics[0] / (disp1 + disparity_change)).float()
        dc_change = depth12/depth1
        sampled_index = torch.from_numpy(sampled_index)
        intrinsics = torch.from_numpy(intrinsics).float()
        flow3d = backproject_flow3d(flow2d, depth1, depth12, intrinsics)

        return image1, image2, depth1, depth2, flow2d, flow3d, intrinsics, sampled_index, dc_change