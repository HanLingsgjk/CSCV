import sys

sys.path.append('core')
import cv2
from glob import glob
import os.path as osp
from PIL import Image
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.util_flow import readPFM
import core.dataset_exp_orin as datasets
from core.utils import frame_utils
from core.utils import flow_viz
from torch.utils.data import DataLoader
from core.raft_cscv import  RAFT343used
from core.utils.utils import InputPadder, forward_interpolate
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#Validation KITTI: 0.935277, 3.001616,log_dc: 40.498310 5000
#Validation KITTI: 0.941944, 3.044805,log_dc: 39.139363 10000
#Validation KITTI: 0.942333, 3.050481,log_dc: 40.371832 11
# scale input depth maps (scaling is undone before evaluation)
DEPTH_SCALE = 1

# exclude pixels with depth > 250
MAX_DEPTH = 250
MIN_DEPTH = 0.05
# exclude extermely fast moving pixels
MAX_FLOW = 250
mesh_grid_cache = {}
def mesh_grid(n, h, w, device, channel_first=True):
    global mesh_grid_cache
    str_id = '%d,%d,%d,%s,%s' % (n, h, w, device, channel_first)
    if str_id not in mesh_grid_cache:
        x_base = torch.arange(0, w, dtype=torch.float32, device=device)[None, None, :].expand(n, h, w)
        y_base = torch.arange(0, h, dtype=torch.float32, device=device)[None, None, :].expand(n, w, h)  # NWH
        grid = torch.stack([x_base, y_base.transpose(1, 2)], 1)  # B2HW
        if not channel_first:
            grid = grid.permute(0, 2, 3, 1)  # BHW2
        mesh_grid_cache[str_id] = grid
    return mesh_grid_cache[str_id]
def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:,None,None].unbind(dim=-1)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = torch.stack([x, y, d], dim=-1)
    return coords


def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape[-2:]

    fx, fy, cx, cy = \
        intrinsics[:, None, None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(),
        torch.arange(wd).to(depths.device).float())

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths
    return torch.stack([X, Y, Z], dim=-1)

def inv_project_2(depths,flow,dc, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape[-2:]

    fx, fy, cx, cy = \
            intrinsics[:, None, None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(),
        torch.arange(wd).to(depths.device).float())
    x = x + flow[0,0, :, :]
    y = y + flow[0,1, :, :]
    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths *dc[:,0]
    return torch.stack([X, Y, Z], dim=-1)

def induced_flowdc(flow2d_est,dc, depth, intrinsics):
    """ Compute 2d and 3d flow fields """

    X0 = inv_project(depth, intrinsics)
    X1 = inv_project_2(depth,flow2d_est,dc,intrinsics)

    flow3d = X1 - X0
    flow2d_est = flow2d_est.permute(0,2,3,1)
    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return flow2d_est, flow3d, valid.float()
def normalize_image(image):
    image = image[:, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=image.device)
    return (image/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

#这个用于测试不同尺度变换区域的误差区别
#分一个，TZ(遮挡区域),FZ(非遮挡区域)，    缩小 (-0.5 - -0.9) (-0.9 - -0.98) (-0.98 - 0.02) (0.2 - 0.1) (0.1 - 0.5)
#讲道理搞一个非遮挡区域的。。。
@torch.no_grad()
def test_scale_change_affect(model):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {'crop_size': 0, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    #datasets.MpiSinteltest(aug_params, split='training', dstype='clean')  datasets.KITTI_test200(split='kitti_test') datasets.FlyingThingsTest()
    test_dataset = datasets.KITTI_test200(split='kitti_test')
    minscale_list, midscale_list,bigscale_list = [], [],[]
    minout_list,midout_list,bigout_list = [], [],[]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    count_ob=0
    count_om = 0
    count_od = 0


    for val_id in range(38, 200, 1):
        #img1, img2, _,_,flow_gt,_,_,_,dc_change= test_dataset[val_id]
        img1, img2, flow_gt, dc_change, _, _, _, _, _, _, _ = test_dataset[val_id]
        #flow_gt = torch.from_numpy(flow_gt)
        #flow_gt =flow_gt.permute(2,0,1)
        image1 = img1[None].cuda()
        image2 = img2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        start.record()
        flow_low, flow_pr,dcout = model(image1, image2, iters=12, test_mode=True)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))


        flow = padder.unpad(flow_pr[0]).cpu()
        dc = padder.unpad(dcout[0]).cpu()
        dcshow = dc.detach().numpy()
        plt.imshow(dcshow[0])
        plt.show()

        logmid = dcshow[0]
        colormap = plt.get_cmap('plasma')
        datamin = np.min(logmid)
        datamax = np.max(logmid)
        mid_data = (datamin + datamax) * 0.5
        lenthmid = 1 / (mid_data - datamin)

        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        cv2.imwrite('%s/%s/dc-%s.jpg' % ('/home/xuxian/RAFT3D/', 'ans', 38), heatmap)

        #计算一下非遮挡掩膜
        input_h = flow_gt.shape[1]
        input_w = flow_gt.shape[2]
        grid = mesh_grid(1, input_h, input_w, device='cpu', channel_first=True)[0].numpy()
        grid_warp = grid + flow.numpy()
        x_out = np.logical_or(grid_warp[0,...] < 0, grid_warp[0,...] > input_w)
        y_out = np.logical_or(grid_warp[1,...] < 0, grid_warp[1,...] > input_h)
        occ_mask = np.logical_or(x_out, y_out).astype(np.uint8)
        flow = padder.unpad(flow_pr[0]).cpu()



        pia = 0.025
        mask_min = (dc_change>(1-pia)) & (dc_change<(1+pia)) & (1-occ_mask)
        mask_mid = (dc_change < (1-pia))  & (dc_change > (0.75-pia)) | ((dc_change >(1+pia))  & (dc_change < (1.25+pia))) & (1-occ_mask)
        mask_big = ((dc_change<=(0.75-pia)) | (dc_change>(1.25+pia))) & (dc_change>0) & (1-occ_mask)

        epe = torch.sum(torch.abs(flow - flow_gt), dim=0)
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        if mask_big.sum() > 0:
            flow_big = epe[mask_big].mean().item()
            #再计算一个外点百分比
            outbig = np.count_nonzero(((epe[mask_big]>3) & ((epe[mask_big] / mag[mask_big])> 0.05)).float())
            count_ob += epe[mask_big].view(-1).shape[0]
            outbigrate = outbig / epe[mask_big].view(-1).shape[0]
        else:
            flow_big = 0
            outbig = 0
            outbigrate = 0

        if mask_mid.sum() > 0:
            flow_mid = epe[mask_mid].mean().item()
            #再计算一个外点百分比
            outmid = np.count_nonzero(((epe[mask_mid]>3) & ((epe[mask_mid] / mag[mask_mid])> 0.05)).float()  )
            count_od += epe[mask_mid].view(-1).shape[0]
            outmidrate = outmid/epe[mask_mid].view(-1).shape[0]
        else:
            flow_mid = 0
            outmid = 0
            outmidrate = 0
        flow_min = epe[mask_min].mean().item()
        outmin = np.count_nonzero(((epe[mask_min]>3) & ((epe[mask_min] / mag[mask_min])> 0.05)).float())
        count_om += epe[mask_min].view(-1).shape[0]
        outminrate = outmin/epe[mask_min].view(-1).shape[0]


        minscale_list.append(flow_min)
        bigscale_list.append(flow_big)
        midscale_list.append(flow_mid)

        minout_list.append(outminrate)
        bigout_list.append(outbigrate)
        midout_list.append(outmidrate)


        print(val_id)

    per_min = count_om/(count_om+count_od+count_ob)
    per_mid = count_od / (count_om + count_od + count_ob)
    per_big = count_ob / (count_om + count_od + count_ob)
    minscale_list = np.array(minscale_list)
    bigscale_list = np.array(bigscale_list)
    midscale_list = np.array(midscale_list)
    print("percent: Scale_min:%f,Scale_mid:%f, Scale_big:%f"%(per_min,per_mid,per_big))

    minout = np.array(minout_list)
    bigout = np.array(bigout_list)
    midout = np.array(midout_list)
    #把哪些不算数的先排除了
    minscale_list_nozero = minout[minout > 0]
    midscale_list_nozero = midout[midout > 0]
    bigscale_list_nozero = bigout[bigout > 0]
    plt.figure(figsize=(5, 3.2))
    plt.boxplot([minscale_list_nozero*100,midscale_list_nozero*100,bigscale_list_nozero*100],
                sym='+',
                whis=1.5,
                widths=0.5,
                labels=['Small','Medium','Large'],
                boxprops={'color': 'blue', 'linewidth': '1.5'},
                medianprops={'color': 'red', 'linewidth': '1.5'},
                whiskerprops={'ls': 'dashed', 'linewidth': '0.5'},
                meanline=True,
                showfliers=False)
    # 添加标题和标签
    #plt.title('Box plot of data1')
    #plt.ylim(-0.02,0.6)
    #plt.xlabel('Value')
    plt.grid(b = True,linewidth = 0.3)
    plt.ylabel('Percentage of optical flow outliers (%)')
    plt.savefig('/home/xuxian/RAFT3D/ans/sintel.pdf',dpi=300, bbox_inches="tight")
    plt.show()




    flowerr_min = np.mean(minscale_list)
    flowerr_mid = np.mean(midscale_list)
    flowerr_big = np.mean(bigscale_list)
    print("Validation KITTI Scale: Scale_min:%f,Scale_mid:%f, Scale_big:%f, out_min:%f, out_mid:%f,out_big:%f" % (flowerr_min,flowerr_mid,flowerr_big,minout,midout,bigout))
    return 1
def prepare_images_and_depths(image1, image2, depth1, depth2, depth_scale=0.2):
    """ padding, normalization, and scaling """

    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image1 = F.pad(image1, [0, pad_w, 0, pad_h], mode='replicate')
    image2 = F.pad(image2, [0, pad_w, 0, pad_h], mode='replicate')
    depth1 = F.pad(depth1[:, None], [0, pad_w, 0, pad_h], mode='replicate')[:, 0]
    depth2 = F.pad(depth2[:, None], [0, pad_w, 0, pad_h], mode='replicate')[:, 0]

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    #image1 = normalize_image(image1.float())
    #image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2, (pad_w, pad_h)
@torch.no_grad()
def test_sceneflow(model):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'drop_last': False}
    train_dataset = datasets.FlyingThingsTest()
    train_loader = DataLoader(train_dataset, **loader_args)
    model.eval()
    count_all, count_sampled = 0, 0
    metrics_all = {'epe2d': 0.0, 'epe3d': 0.0, '1px': 0.0, '5cm': 0.0, '10cm': 0.0}
    metrics_flownet3d = {'epe2d': 0.0,'1px': 0.0,'epe3d': 0.0, '5cm': 0.0, '10cm': 0.0}
    sumnum = 0
    for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
        sumnum  =sumnum+1
        #if sumnum>600:
         #   break
        image1, image2, depth1, depth2, flow2d, flow3d, intrinsics, index,_ = \
            [data_item.cuda() for data_item in test_data_blob]

        mag = torch.sum(flow2d ** 2, dim=-1).sqrt()
        valid = (mag.reshape(-1) < MAX_FLOW) & (depth1.reshape(-1) < MAX_DEPTH)

        # pad and normalize images
        image1, image2, depth1, depth2, padding = \
            prepare_images_and_depths(image1, image2, depth1, depth2, DEPTH_SCALE)

        # run the model
        _, flow_pr,dc = model(image1, image2, iters=12,test_mode=True)

        # use transformation field to extract 2D and 3D flow
        flow2d_est, flow3d_est, _ = induced_flowdc(flow_pr,dc, depth1, intrinsics)

        # unpad the flow fields / undo depth scaling
        flow2d_est = flow2d_est[:, :-4, :, :2]
        flow3d_est = flow3d_est[:, :-4] / DEPTH_SCALE

        epe2d = torch.sum((flow2d_est - flow2d) ** 2, -1).sqrt()
        epe3d = torch.sum((flow3d_est - flow3d) ** 2, -1).sqrt()

        # our evaluation (use all valid pixels)
        epe2d_all = epe2d.reshape(-1)[valid].double().cpu().numpy()
        epe3d_all = epe3d.reshape(-1)[valid].double().cpu().numpy()

        count_all += epe2d_all.shape[0]
        metrics_all['epe2d'] += epe2d_all.sum()
        metrics_all['epe3d'] += epe3d_all.sum()
        metrics_all['1px'] += np.count_nonzero(epe2d_all < 1.0)
        metrics_all['5cm'] += np.count_nonzero(epe3d_all < .05)
        metrics_all['10cm'] += np.count_nonzero(epe3d_all < .10)

        # FlowNet3D evaluation (only use sampled non-occ pixels)
        epe3d = epe3d[0, index[0, 0], index[0, 1]]
        epe2d = epe2d[0, index[0, 0], index[0, 1]]

        epe2d_sampled = epe2d.reshape(-1).double().cpu().numpy()
        epe3d_sampled = epe3d.reshape(-1).double().cpu().numpy()

        metrics_flownet3d['epe2d'] += epe2d_sampled.sum()
        count_sampled += epe2d_sampled.shape[0]
        metrics_flownet3d['1px'] += np.count_nonzero(epe2d_sampled < 1.0)
        metrics_flownet3d['epe3d'] += epe3d_sampled.mean()
        metrics_flownet3d['5cm'] += (epe3d_sampled < .05).astype(np.float).mean()
        metrics_flownet3d['10cm'] += (epe3d_sampled < .10).astype(np.float).mean()

    # Average results over all valid pixels
    print("all...")
    for key in metrics_all:
        print(key, metrics_all[key] / count_all)
    metrics_flownet3d['epe2d'] = (i_batch + 1)*metrics_flownet3d['epe2d']/count_sampled
    metrics_flownet3d['1px'] = (i_batch + 1)*metrics_flownet3d['1px'] / count_sampled
    # FlowNet3D evaluation methodology
    print("non-occ (FlowNet3D Evaluation)...")
    for key in metrics_flownet3d:
        print(key, metrics_flownet3d[key] / (i_batch + 1))

@torch.no_grad()
def test_optical_flow(model):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'drop_last': False}
    train_dataset = datasets.FlyingThingsTest()
    train_loader = DataLoader(train_dataset, **loader_args)
    model.eval()
    count_all, count_sampled = 0, 0
    metrics_flownet3d = {'epe2d': 0.0,'1px': 0.0,'epe3d': 0.0, '5cm': 0.0, '10cm': 0.0}

    for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
        image1, image2, depth1, depth2, flow2d, flow3d, intrinsics, index = \
            [data_item.cuda() for data_item in test_data_blob]


        # pad and normalize images
        image1, image2, depth1, depth2, padding = \
            prepare_images_and_depths(image1, image2, depth1, depth2, DEPTH_SCALE)

        # run the model
        _, flow2d_est = model(image1, image2, iters=12,test_mode=True)

        # unpad the flow fields / undo depth scaling
        flow2d_est = flow2d_est.permute(0,2,3,1)[:, :-4, :, :2]

        epe2d = torch.sum((flow2d_est - flow2d) ** 2, -1).sqrt()

        # FlowNet3D evaluation (only use sampled non-occ pixels)
        epe2d = epe2d[0, index[0, 0], index[0, 1]]

        epe2d_sampled = epe2d.reshape(-1).double().cpu().numpy()

        metrics_flownet3d['epe2d'] += epe2d_sampled.sum()
        count_sampled += epe2d_sampled.shape[0]
        metrics_flownet3d['1px'] += np.count_nonzero(epe2d_sampled < 1.0)

    # Average results over all valid pixels
    print("all...")
    metrics_flownet3d['epe2d'] = (i_batch + 1)*metrics_flownet3d['epe2d']/count_sampled
    metrics_flownet3d['1px'] = (i_batch + 1)*metrics_flownet3d['1px'] / count_sampled
    # FlowNet3D evaluation methodology
    print("non-occ (FlowNet3D Evaluation)...")
    for key in metrics_flownet3d:
        print(key, metrics_flownet3d[key] / (i_batch + 1))

@torch.no_grad()
def create_sintel_submission(model, iters=16, warm_start=False, output_path='/home/xuxian/RAFT3D/sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr, dc = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)

            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_dirshow = '/home/xuxian/RAFT3D/sintel_show/flow0/'
            output_dirshow = os.path.join(output_dirshow, dstype, sequence)
            if not os.path.exists(output_dirshow):
                os.makedirs(output_dirshow)
            output_fileshow = os.path.join(output_dirshow, 'frame%04d.png' % (frame + 1))
            plt.imshow(flow[:, :, 0])
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig(output_fileshow)
            plt.clf()
            print(sequence+'frame%04d.png' %frame)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence
@torch.no_grad()
def create_ans_show(model, iters=12, output_path='bangqiushow'):
    #/home/xuxian/two_steam/jpegs_256/#submitsubmitother
    model.eval()
    test_dataset = datasets.KITTI(split='submitother', aug_params=None,root='/home/xuxian/RAFT3D/input/2011_10_03_drive_0047_sync/image_02/data/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr,dc = model(image1, image2, iters=iters, test_mode=True)
        dc = dc.log()
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        dc = padder.unpad(dc[0]).permute(1, 2, 0).cpu().numpy()
        image1 = padder.unpad(image1[0]).permute(1, 2, 0).cpu().numpy()


        flo = flow_viz.flow_to_image(flow)

        logmid = np.power(10,dc[:,:,0])
        colormap = plt.get_cmap('plasma')
        datamin = np.min(logmid)
        datamax = np.max(logmid)
        mid_data = (datamin+datamax)*0.5
        lenthmid =1/(mid_data-datamin)
        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        #plt.imshow(heatmap)
        #plt.show()

        #pic_save = np.concatenate((heatmap, image1), axis=0)
        #cv2.imwrite('%s/%s/pic-%s.jpg' % ('/home/lh/RAFT3D-DEPTH', output_path, test_id), pic_save)

        flow_tau  = np.concatenate((image1, flo,heatmap), axis=0)
        cv2.imwrite('%s/%s/tau_flow-%s.jpg' % ('/home/xuxian/RAFT3D/', output_path, test_id), flow_tau)

        print(test_id)
#用于kitti15 senceflow test
@torch.no_grad()
def create_ucf101_submission(model, iters=12, output_path='bq_submit',start = 2000):
    """ 用于提取动作检测数据集的三维光流 """
    #/home/xuxian/two_steam/jpegs_256/
    model.eval()
    path_root = '/home/xuxian/RAFT3D/input/pic_out/'
    lst = os.listdir(path_root)
    for idx in range(len(lst)):
        name = lst[idx]
        rootuse = path_root+name
        test_dataset = datasets.KITTI(split='submitother', aug_params=None,root=rootuse+'/')
        pathsave = '%s/%s/bq%s/' % ('/home/xuxian/RAFT3D/', output_path, name)
        isExists = os.path.exists(pathsave)
        print(str(idx)+'/'+str(len(lst)))
        if not isExists:
            os.makedirs(pathsave)

            for test_id in range(len(test_dataset)):
                image1, image2, (frame_id,) = test_dataset[test_id]
                padder = InputPadder(image1.shape, mode='kitti')
                image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

                _, flow_pr,dc = model(image1, image2, iters=iters, test_mode=True)

                flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
                dc = padder.unpad(dc[0]).permute(1, 2, 0).cpu().numpy()

                flo = flow_viz.flow_to_image(flow)
                image1 = padder.unpad(image1[0]).permute(1, 2, 0).cpu().numpy()
                image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

                logmid = dc[:,:,0]
                colormap = plt.get_cmap('binary')
                logmid = ((logmid - 1) * 20).clip(-1, 1) * 128 + 128
                heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
                #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                frame_idx = 'frame' + str(test_id).zfill(6)

                flow_tau = np.concatenate((image1, flo, heatmap), axis=0)

                cv2.imwrite('%s/%s/bq%s/%s.jpg' % ('/home/xuxian/RAFT3D', output_path,name, frame_idx), flow_tau)
                print('%s/%s/bq%s/%s.jpg' % ('/home/xuxian/RAFT3D', output_path,name, frame_idx))
        else:
            print('go')
def kitti_sceneflow_submission(model, iters=12, output_path=''):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    #/home/xuxian/RAFT3D/datasets/testing/
    #/home/lh/RAFT_master/dataset/kitti_scene/testing
    test_dataset = datasets.KITTI(split='test', aug_params=None,root='/new_data/kitti_data/datasets/testing')

    for test_id in range(0,len(test_dataset),1):
        image1, image2, (frame_id,),disp = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        _, flow_pr,dc = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).detach().cpu().numpy()
        dc = padder.unpad(dc[0]).permute(1, 2, 0).detach().cpu().numpy()
        disp2 = disp/dc[:,:,0]
        '''
        plt.imshow(flow[:,:,0])
        plt.show()
        plt.clf()
        '''
        disp1 =  (disp * 256).astype('uint16')
        disp2 =  (disp2 * 256).astype('uint16')
        output_filename = os.path.join('/home/xuxian/RAFT3D/submit/flow/', frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        cv2.imwrite('%s/%s' % ('/home/xuxian/RAFT3D/submit/disp_0', frame_id), disp1)
        cv2.imwrite('%s/%s' % ('/home/xuxian/RAFT3D/submit/disp_1', frame_id), disp2)
        print(test_id)
# 这里顺便搞一个测评函数,用于160-40测评
#这里写一个数据集生成函数
def kitti_testdata_get():
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='training',get_depth=1)
    test_dataset.k = 0
    test_dataset.kr = 0
    for test_id in range(0, len(test_dataset), 5):
        img1,img2,flow,dc_change,d1,d2,disp1,disp2,mask,frame_id = test_dataset[test_id]
        flow = flow.permute(1, 2, 0).detach().cpu().numpy()
        img1 = img1.permute(1, 2, 0).detach().cpu().numpy()
        img2 = img2.permute(1, 2, 0).detach().cpu().numpy()
        imgm = np.zeros_like(img1)

        imgm[:, :, 0] = img1[:, :, 0]
        imgm[:, :, 2] = img1[:, :, 2]
        img1[:, :, 0] = imgm[:, :, 2]
        img1[:, :, 2] = imgm[:, :, 0]

        imgm[:, :, 0] = img2[:, :, 0]
        imgm[:, :, 2] = img2[:, :, 2]
        img2[:, :, 0] = imgm[:, :, 2]
        img2[:, :, 2] = imgm[:, :, 0]

        disp1 = (disp1 * 256).astype('uint16')
        disp2 = (disp2 * 256).astype('uint16')
        mask2 = mask.astype('uint8')
        frame_id = frame_id[0]
        output_filename = os.path.join('/home/lh/RAFT3D-DEPTH/data_train_test/flow/', frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)
        cv2.imwrite('%s/%s' % ('/home/lh/RAFT3D-DEPTH/data_train_test/img1', frame_id), img1)
        cv2.imwrite('%s/%s' % ('/home/lh/RAFT3D-DEPTH/data_train_test/img2', frame_id), img2)
        cv2.imwrite('%s/%s' % ('/home/lh/RAFT3D-DEPTH/data_train_test/disp_0', frame_id), disp1)
        cv2.imwrite('%s/%s' % ('/home/lh/RAFT3D-DEPTH/data_train_test/disp_1', frame_id), disp2)
        cv2.imwrite('%s/%s' % ('/home/lh/RAFT3D-DEPTH/data_train_test/mask', frame_id), mask2)
        print(test_id)
    return 0
@torch.no_grad()
def validate_kitti_test(model, iters=12):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    test_dataset = datasets.KITTI_test(split='kitti_test')#kitti_test
    out_b_list, epe_b_list = [], []
    out_f_list, epe_f_list = [], []
    dc_b_list = []
    dc_f_list = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for val_id in range(0, len(test_dataset), 1):
        img1, img2, flow_gt, dc_change, d1, d2, disp1, disp2, mask,_, frame_id = test_dataset[val_id]
        image1 = img1[None].cuda()
        image2 = img2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        start.record()
        flow_low, flow_pr,dchange= model(image1, image2, iters=iters, test_mode=True)
        #dchange = flow_pr[:,0:1,:,:]
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))

        flow = padder.unpad(flow_pr[0]).cpu()
        dchange = padder.unpad(dchange[0]).cpu()

        dc_change = torch.from_numpy(dc_change).float()
        mask = torch.from_numpy(mask)
        dc_s = dchange[0].detach().cpu().numpy()
        '''
        plt.imshow(dc_s)
        plt.show()
        fl_s = flow[0].detach().cpu().numpy()
        plt.imshow(fl_s)
        plt.show()
        '''
        gt_dc = dc_change
        logmiderr_b = torch.abs(dchange.log() - gt_dc.log())[0][mask == 1].mean()
        logmiderr_f = torch.abs(dchange.log() - gt_dc.log())[0][mask == 2].mean()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()


        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)

        val_b = mask.view(-1) == 1
        val_f = mask.view(-1) == 2
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_b_list.append(epe[val_b].mean().item())
        out_b_list.append(out[val_b].cpu().numpy())

        epe_f_list.append(epe[val_f].mean().item())
        out_f_list.append(out[val_f].cpu().numpy())

        dc_b_list.append(logmiderr_b.item())
        dc_f_list.append(logmiderr_f.item())
        print(val_id)
    epe_b_list = np.array(epe_b_list)
    epe_f_list = np.array(epe_f_list)

    out_b_list = np.concatenate(out_b_list)
    out_f_list = np.concatenate(out_f_list)

    dc_b_list = np.array(dc_b_list)
    dc_f_list = np.array(dc_f_list)

    dc_b = np.mean(dc_b_list) * 10000
    dc_f = np.mean(dc_f_list) * 10000
    epe_b = np.mean(epe_b_list)
    epe_f = np.mean(epe_f_list)
    f1_b = 100 * np.mean(out_b_list)
    f1_f = 100 * np.mean(out_f_list)
    print("Validation KITTI: eb:%f, ef:%f, f1b:%f, f1f:%f, log_dc_b: %f, log_dc_f: %f" % (epe_b,epe_f, f1_b, f1_f,dc_b,dc_f))
    return {'kitti-epe': epe_b, 'kitti-f1': f1_b}
@torch.no_grad()
def validate_kitti(model, iters=12):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='testing')
    val_dataset.k = 0
    val_dataset.kitti_test=1
    out_list, epe_list = [], []
    dc_list = []
    for val_id in range(0,len(val_dataset),5):
        image1, image2, flow_gt,dc_change,d1,d2,disp1,disp2,mask,valid_gt,_ = val_dataset[val_id]

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        image1 = image1.unsqueeze(0).cuda()
        image2 = image2.unsqueeze(0).cuda()
        flow_low, flow_pr ,dchange= model(image1, image2, iters=iters, test_mode=True)

        flow = padder.unpad(flow_pr[0]).cpu()
        dchange = padder.unpad(dchange[0]).cpu()
        maskdc = dc_change!=0
        gt_dc = torch.from_numpy(dc_change[:, :])
        valid_gt = torch.from_numpy(valid_gt)
        logmiderr = torch.abs(dchange.log()-gt_dc.log())[0][maskdc>0].mean()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        dc_list.append(logmiderr.item())
        print(val_id)
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    dc_list  = np.array(dc_list)
    dc = np.mean(dc_list)*10000
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f,log_dc: %f" % (epe, f1,dc))
    return {'kitti-epe': epe, 'kitti-f1': f1}
@torch.no_grad()
def validate_driving(model, iters=12):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.Driving(split='training')

    out_list, epe_list = [], []
    dc_list = []
    for val_id in range(0,len(val_dataset),100):
        image1, image2, flow_gt,dc_change, valid_gt = val_dataset[val_id,0]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr ,dchange= model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()
        dchange = padder.unpad(dchange[0]).cpu()
        maskdc = dc_change[1,:,:]
        gt_dc = dc_change[0, :, :]
        logmiderr = torch.abs(dchange.log()-gt_dc.log())[0][maskdc>0].mean()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        dc_list.append(logmiderr.item())
        print(val_id)
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    dc_list  = np.array(dc_list)
    dc = np.mean(dc_list)*10000
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation Driving: %f, %f,log_dc: %f" % (epe, f1,dc))
    return {'driving-epe': epe, 'driving-f1': f1}
#测试TTC，kitti默认间隔是1S  0.1/（1-tao）
@torch.no_grad()
def validate_kitti_TTC_test(model, iters=12):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    test_dataset = datasets.KITTI_test(split='kitti_test')
    t1_list = []
    t2_list = []
    t5_list = []
    for val_id in range(0, len(test_dataset), 1):
        img1, img2, flow_gt, dc_change, d1, d2, disp1, disp2, mask, valid_gt,frame_id = test_dataset[val_id]
        image1 = img1[None].cuda()
        image2 = img2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr, dchange = model(image1, image2, iters=iters, test_mode=True)
        '''
        flow = padder.unpad(flow_pr[0]).cpu()
        dchange = padder.unpad(dchange[0]).cpu()

        dc_change = torch.from_numpy(dc_change).float()
        mask = torch.from_numpy(mask)
        gt_dc = dc_change
        #计算TTC
        ttcmask = dchange>=1
        ttcmask_gt = gt_dc >= 1

        ttc = 0.1/(1-dchange)
        ttc_gt = 0.1/(1-dc_change)
        ttc[ttcmask] = 1000000
        ttc_gt[ttcmask_gt] = 1000000

        ttc1 = (ttc < 1)[0]
        ttc2 = (ttc < 2)[0]
        ttc5 = (ttc < 5)[0]

        ttc1gt = ttc_gt < 1
        ttc2gt = ttc_gt < 2
        ttc5gt = ttc_gt < 5

        t1 = torch.logical_xor(ttc1,ttc1gt)[mask].sum() / mask.sum()
        t2 = torch.logical_xor(ttc2, ttc2gt)[mask].sum() / mask.sum()
        t5 = torch.logical_xor(ttc5, ttc5gt)[mask].sum() / mask.sum()

        t1_list.append(t1)
        t2_list.append(t2)
        t5_list.append(t5)

    t1_list = np.array(t1_list)
    t2_list = np.array(t2_list)
    t5_list = np.array(t5_list)

    print("Validation KITTI: err_1s:%f, err_2s:%f, err_5s:%f" % (t1_list.mean(),t2_list.mean(),t5_list.mean()))
    '''
    return 0
#测试Binary_TTC，kitti默认间隔是1S  0.1/（1-tao）
@torch.no_grad()
def validate_kitti_BTTC_test():
    """ Peform validation using the KITTI-2015 (train) split """
    test_dataset = datasets.KITTI_test(split='kitti_test')
    t1_list = []
    t2_list = []
    t5_list = []

    for val_id in range(0, len(test_dataset), 1):
        img1, img2, flow_gt, dc_change, d1, d2, disp1, disp2, mask,_, frame_id = test_dataset[val_id]

        s1_path = os.path.join('/home/xuxian/RAFT3D/res/0',str(val_id)+'.png')
        s2_path = os.path.join('/home/xuxian/RAFT3D/res/1', str(val_id)+ '.png')
        s5_path = os.path.join('/home/xuxian/RAFT3D/res/2', str(val_id)+ '.png')
        dc_change = torch.from_numpy(dc_change).float()
        mask = torch.from_numpy(mask)
        gt_dc = dc_change
        #计算TTC
        ttcmask_gt = gt_dc >= 1

        ttc_gt = 0.1/(1-dc_change)
        ttc_gt[ttcmask_gt] = 1000000

        s1 = torch.from_numpy(np.array(frame_utils.read_gen(s1_path)))
        s2 = torch.from_numpy(np.array(frame_utils.read_gen(s2_path)))
        s5 = torch.from_numpy(np.array(frame_utils.read_gen(s5_path)))
        ttc1 = (s1 >100)
        ttc2 = (s2 >100)
        ttc5 = (s5 >100)

        ttc1gt = ttc_gt < 1
        ttc2gt = ttc_gt < 2
        ttc5gt = ttc_gt < 5

        t1 = torch.logical_xor(ttc1,ttc1gt)[mask].sum() / mask.sum()
        t2 = torch.logical_xor(ttc2, ttc2gt)[mask].sum() / mask.sum()
        t5 = torch.logical_xor(ttc5, ttc5gt)[mask].sum() / mask.sum()

        t1_list.append(t1)
        t2_list.append(t2)
        t5_list.append(t5)
        print(val_id)
    t1_list = np.array(t1_list)
    t2_list = np.array(t2_list)
    t5_list = np.array(t5_list)

    print("Validation KITTI: err_1s:%f, err_2s:%f, err_5s:%f" % (t1_list.mean(),t2_list.mean(),t5_list.mean()))
    return 0
#测试expansion TTC，kitti默认间隔是1S  0.1/（1-tao）
@torch.no_grad()
def validate_kitti_exTTC_test():
    """ Peform validation using the KITTI-2015 (train) split """
    test_dataset = datasets.KITTI_test(split='kitti_test')
    t1_list = []
    t2_list = []
    t5_list = []
    mid_path = sorted(glob('/home/xuxian/RAFT3D/mid/*_10.pfm'))
    for val_id in range(0, len(test_dataset), 1):
        img1, img2, flow_gt, dc_change, d1, d2, disp1, disp2, mask, _,frame_id = test_dataset[val_id]
        mid = readPFM(mid_path[val_id])[0]
        dchange = torch.from_numpy(mid.copy()).exp()
        dc_change = torch.from_numpy(dc_change).float()
        mask = torch.from_numpy(mask)
        gt_dc = dc_change
        #计算TTC
        ttcmask = dchange>=1
        ttcmask_gt = gt_dc >= 1
        ttcmask_gt1 = gt_dc < 1
        ttc = 0.1/(1-dchange)
        ttc_gt = 0.1/(1-dc_change)
        ttc[ttcmask] = 1000000
        ttc_gt[ttcmask_gt] = 1000000

        ttc1 = (ttc < 1)
        ttc2 = (ttc < 2)
        ttc5 = (ttc < 5)

        ttc1gt = ttc_gt < 1
        ttc2gt = ttc_gt < 2
        ttc5gt = ttc_gt < 5


        t1 = torch.logical_xor(ttc1,ttc1gt)[mask].sum() / mask.sum()
        t2 = torch.logical_xor(ttc2, ttc2gt)[mask].sum() / mask.sum()
        t5 = torch.logical_xor(ttc5, ttc5gt)[mask].sum() / mask.sum()

        t1_list.append(t1)
        t2_list.append(t2)
        t5_list.append(t5)

    t1_list = np.array(t1_list)
    t2_list = np.array(t2_list)
    t5_list = np.array(t5_list)

    print("Validation KITTI: err_1s:%f, err_2s:%f, err_5s:%f" % (t1_list.mean(),t2_list.mean(),t5_list.mean()))
    return 0
#测试ojsf TTC，kitti默认间隔是1S  0.1/（1-tao）
@torch.no_grad()
def validate_kitti_osTTC_test():
    """ Peform validation using the KITTI-2015 (train) split """
    test_dataset = datasets.KITTI_test(split='kitti_test')
    t1_list = []
    t2_list = []
    t5_list = []
    d0_path = sorted(glob('/home/xuxian/RAFT3D/datasets/disp40_0/*.png'))
    d1_path = sorted(glob('/home/xuxian/RAFT3D/datasets/disp40_1/*.png'))
    for val_id in range(0, len(test_dataset), 1):
        img1, img2, flow_gt, dc_change, d1, d2, disp1, disp2, mask, _,frame_id = test_dataset[val_id]
        d0_os = torch.from_numpy(np.array(frame_utils.read_gen(d0_path[val_id])))
        d1_os = torch.from_numpy(np.array(frame_utils.read_gen(d1_path[val_id])))
        mid = d0_os/d1_os
        dchange =mid
        dc_change = torch.from_numpy(dc_change).float()
        mask = torch.from_numpy(mask)
        gt_dc = dc_change
        #计算TTC
        ttcmask = dchange>=1
        ttcmask_gt = gt_dc >= 1
        ttcmask_gt1 = gt_dc < 1
        ttc = 0.1/(1-dchange)
        ttc_gt = 0.1/(1-dc_change)
        ttc[ttcmask] = 1000000
        ttc_gt[ttcmask_gt] = 1000000

        ttc1 = (ttc < 1)
        ttc2 = (ttc < 2)
        ttc5 = (ttc < 5)

        ttc1gt = ttc_gt < 1
        ttc2gt = ttc_gt < 2
        ttc5gt = ttc_gt < 5


        t1 = torch.logical_xor(ttc1,ttc1gt)[mask].sum() / mask.sum()
        t2 = torch.logical_xor(ttc2, ttc2gt)[mask].sum() / mask.sum()
        t5 = torch.logical_xor(ttc5, ttc5gt)[mask].sum() / mask.sum()

        t1_list.append(t1)
        t2_list.append(t2)
        t5_list.append(t5)

    t1_list = np.array(t1_list)
    t2_list = np.array(t2_list)
    t5_list = np.array(t5_list)

    print("Validation KITTI: err_1s:%f, err_2s:%f, err_5s:%f" % (t1_list.mean(),t2_list.mean(),t5_list.mean()))
    return 0
#这个是用来画ours论文里面的图的
@torch.no_grad()
def validate_kitti_and_plot(model, iters=12):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='testing')
    val_dataset.k = 0
    val_dataset.kitti_test=1
    out_list, epe_list = [], []
    dc_list = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for val_id in range(0,len(val_dataset),5):
        image1, image2, flow_gt,dc_change,d1,d2,disp1,disp2,mask,valid_gt,_ = val_dataset[val_id]
        #error map
        error_map = np.zeros([image1.shape[1],image1.shape[2],3])

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        image1 = image1.unsqueeze(0).cuda()
        image2 = image2.unsqueeze(0).cuda()
        start.record()
        flow_low, flow_pr ,dchange= model(image1, image2, iters=iters, test_mode=True)
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))

        flow = padder.unpad(flow_pr[0]).cpu()
        dchange = padder.unpad(dchange[0]).cpu().numpy()

        maskdc = dc_change!=0
        maskd = dc_change==0
        gt_dc = dc_change[:, :]
        logmiderr = (abs(np.log(dchange[0])-np.log(gt_dc)))*[maskdc]
        logmiderr[0, maskd] = 0
        logmiderr_print = np.mean((abs(np.log(dchange[0]) - np.log(gt_dc)))[maskdc])
        error_map[:,:,0] =logmiderr*2000
        error_map[:, :,1] = logmiderr*2000
        error_map[:, :,2] = logmiderr*2000
        imgname = str(val_id)+'_'+str(logmiderr_print)
        #画errror_map
        cv2.imwrite('%s/%s/pic-%s.jpg' % ('/home/xuxian/RAFT3D','error_map_ours', imgname), error_map)
        #画dc_change
        logmid = dchange[0]
        colormap = plt.get_cmap('plasma')
        datamin = np.min(logmid)
        datamax = np.max(logmid)
        mid_data = (datamin + datamax) * 0.5
        lenthmid = 1 / (mid_data - datamin)

        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        cv2.imwrite('%s/%s/dc-%s.jpg' % ('/home/xuxian/RAFT3D', 'error_map_ours', imgname), heatmap)

        # 画gt_dc_change
        logmid = dc_change
        colormap = plt.get_cmap('plasma')
        datamin = np.min(logmid[maskdc])
        datamax = np.max(logmid[maskdc])
        mid_data = (datamin + datamax) * 0.5
        lenthmid = 1 / (mid_data - datamin)

        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        heatmap[maskd,:]=255
        cv2.imwrite('%s/%s/gt_dc-%s.jpg' % ('/home/lh/RAFT3D-DEPTH', 'error_map_ours', imgname), heatmap)
    return 0
#这个是用来画ours论文里面expansion的图的
@torch.no_grad()
def validate_exkitti_and_plot(model, iters=12):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='testing')
    val_dataset.k = 0
    val_dataset.kitti_test=1
    out_list, epe_list = [], []
    dc_list = []
    mid_path = sorted(glob('/home/xuxian/RAFT3D/optical_expansion_test_mid/oemid/*_10.pfm'))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for val_id in range(0,len(val_dataset),5):
        image1, image2, flow_gt,dc_change,d1,d2,disp1,disp2,mask,valid_gt,_ = val_dataset[val_id]
        #error map
        error_map = np.zeros([image1.shape[1],image1.shape[2],3])

        mid = readPFM(mid_path[int(val_id/5)])[0]
        dchange = torch.from_numpy(mid.copy()).exp()
        dc_change = torch.from_numpy(dc_change).float()
        dchange = dchange.numpy()
        dc_change = dc_change.numpy()


        maskdc = dc_change!=0
        maskd = dc_change==0
        gt_dc = dc_change[:, :]
        logmiderr = (abs(np.log(dchange)-np.log(gt_dc)))*[maskdc]
        logmiderr[0, maskd] = 0
        logmiderr_print = np.mean((abs(np.log(dchange) - np.log(gt_dc)))[maskdc])
        error_map[:,:,0] =logmiderr*2000
        error_map[:, :,1] = logmiderr*2000
        error_map[:, :,2] = logmiderr*2000
        imgname = str(val_id)+'_'+str(logmiderr_print)
        #画errror_map
        cv2.imwrite('%s/%s/pic-%s.jpg' % ('/home/xuxian/RAFT3D','error_map_ex', imgname), error_map)
        #画dc_change
        logmid = dchange
        colormap = plt.get_cmap('plasma')
        datamin = np.min(logmid)
        datamax = np.max(logmid)
        mid_data = (datamin + datamax) * 0.5
        lenthmid = 1 / (mid_data - datamin)

        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        cv2.imwrite('%s/%s/dc-%s.jpg' % ('/home/xuxian/RAFT3D', 'error_map_ex', imgname), heatmap)

        # 画gt_dc_change
        logmid = dc_change
        colormap = plt.get_cmap('plasma')
        datamin = np.min(logmid[maskdc])
        datamax = np.max(logmid[maskdc])
        mid_data = (datamin + datamax) * 0.5
        lenthmid = 1 / (mid_data - datamin)

        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        heatmap[maskd,:]=255
        cv2.imwrite('%s/%s/gt_dc-%s.jpg' % ('/home/xuxian/RAFT3D', 'error_map_ex', imgname), heatmap)
    return 0
#这个是用来画ours论文里面OSF的图的
@torch.no_grad()
def validate_osfkitti_and_plot(model, iters=12):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    d0_path = sorted(glob('/home/xuxian/RAFT3D/datasets/disp40_0/*.png'))
    d1_path = sorted(glob('/home/xuxian/RAFT3D/datasets/disp40_1/*.png'))
    val_dataset = datasets.KITTI(split='testing')
    val_dataset.k = 0
    val_dataset.kitti_test=1
    out_list, epe_list = [], []
    dc_list = []
    mid_path = sorted(glob('/home/xuxian/RAFT3D/optical_expansion_test_mid/oemid/*_10.pfm'))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for val_id in range(0,len(val_dataset),5):
        image1, image2, flow_gt,dc_change,d1,d2,disp1,disp2,mask,valid_gt,_ = val_dataset[val_id]
        #error map
        error_map = np.zeros([image1.shape[1],image1.shape[2],3])

        d0_os = (np.array(frame_utils.read_gen(d0_path[int(val_id/5)])))
        d1_os = (np.array(frame_utils.read_gen(d1_path[int(val_id/5)])))
        mid = d0_os / d1_os

        dchange = torch.from_numpy(mid.copy())
        dc_change = torch.from_numpy(dc_change).float()
        dchange = dchange.numpy()
        dc_change = dc_change.numpy()


        maskdc = dc_change!=0
        maskd = dc_change==0
        gt_dc = dc_change[:, :]
        logmiderr = (abs(np.log(dchange)-np.log(gt_dc)))*[maskdc]
        logmiderr[0, maskd] = 0
        logmiderr_print = np.mean(np.clip((abs(np.log(dchange) - np.log(gt_dc))),-1,1)[maskdc])
        error_map[:,:,0] =np.clip(logmiderr*2000,0,255)
        error_map[:, :,1] =np.clip(logmiderr*2000,0,255)
        error_map[:, :,2] =np.clip(logmiderr*2000,0,255)
        imgname = str(val_id)+'_'+str(logmiderr_print)
        #画errror_map
        cv2.imwrite('%s/%s/pic-%s.jpg' % ('/home/xuxian/RAFT3D','error_map_osf', imgname), error_map)
        #画dc_change
        logmid = dchange
        colormap = plt.get_cmap('plasma')
        datamin = 0.74638#np.min(logmid[maskdc])
        datamax = 1.02#np.max(logmid[maskdc])
        sc = plt.imshow(dchange, vmin=0.74638, vmax=1.02, cmap=plt.cm.plasma)  # 限制范围为0-100
        plt.colorbar()
        plt.show()
        mid_data = (datamin + datamax) * 0.5
        lenthmid = 1 / (mid_data - datamin)

        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        cv2.imwrite('%s/%s/dc-%s.jpg' % ('/home/xuxian/RAFT3D', 'error_map_osf', imgname), heatmap)

        # 画gt_dc_change
        logmid = dc_change
        colormap = plt.get_cmap('plasma')
        datamin = np.min(logmid[maskdc])
        datamax = np.max(logmid[maskdc])
        mid_data = (datamin + datamax) * 0.5
        lenthmid = 1 / (mid_data - datamin)

        logmid = ((logmid - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
        heatmap = (colormap((logmid).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        heatmap[maskd,:]=255
        cv2.imwrite('%s/%s/gt_dc-%s.jpg' % ('/home/xuxian/RAFT3D', 'error_map_osf', imgname), heatmap)
    return 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--start', default=0, type=int,
                        help='where to start')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT343used(args))

    pretrained_dict = torch.load(args.model)
    old_list = {}
    for k, v in pretrained_dict.items():
        # if k.find('encoder.convc1')<0 :
        old_list.update({k: v})
    model.load_state_dict(old_list, strict=False)

    model.cuda()
    model.eval()
    #validate_kitti_test(model.module)
    #validate_kitti_BTTC_test()
    #validate_kitti_exTTC_test()
    #validate_kitti_osTTC_test()
    #validate_kitti_TTC_test(model.module)
    with torch.no_grad():
        #test_optical_flow(model)
        #test_sceneflow(model)
        #create_sintel_submission(model.module)
        #validate_kitti_and_plot(model.module)
        #validate_kitti_TTC_test(model.module)
        validate_kitti(model)
        #kitti_testdata_get()
        #test_scale_change_affect(model.module)
        #kitti_sceneflow_submission(model.module)
        #create_sintel_submission(model.module, warm_start=True)
        #create_ans_show(model.module)
        #create_ucf101_submission(model.module)

         #if args.dataset == 'kitti':
           # validate_kitti(model.module)
