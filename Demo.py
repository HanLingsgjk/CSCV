import sys

sys.path.append('core')
import cv2
from glob import glob
import os.path as osp
from PIL import Image
import argparse
import os
import numpy as np
from core.utils.flow_viz import flow2rgb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.utils import frame_utils
from core.raft_cscv import  RAFT343used
from core.utils.utils import InputPadder
def Davis_demo(model,path1,path2,outpath, iters=12):
    img1 = frame_utils.read_gen(path1)
    img2 = frame_utils.read_gen(path2)

    pathsplit = path1.split('/')
    idout = pathsplit[-1].split('.')[0]

    img1 = np.array(img1).astype(np.uint8)[..., :3]
    img2 = np.array(img2).astype(np.uint8)[..., :3]
    img1 = cv2.resize(img1, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    image1 = img1[None].cuda()
    image2 = img2[None].cuda()
    padder = InputPadder(image1.shape, mode='kitti',sp=16)
    image1, image2 = padder.pad(image1, image2)

    #res = gma_forward(image1, image2)
    #flow_pr = res['flow_preds'][0]
    flow_low, flow_pr, dchange = model(image1, image2, iters=iters, test_mode=True)
    flow = padder.unpad(flow_pr[0]).detach().cpu()


    dchange = padder.unpad(dchange[0,0]).detach().cpu().numpy()
    frame_id = idout + 'depth_change.png'
    datamin = np.min(dchange)
    datamax = np.max(dchange)
    mid_data = (datamin + datamax) * 0.5
    lenthmid = 1 / (mid_data - datamin)
    dchange = ((dchange - mid_data) * lenthmid).clip(-1, 1) * 128 + 128
    # dchange = ((dchange - 1)*16).clip(-1, 1) * 128 + 128
    colormap = plt.get_cmap('plasma')  # plasma viridis
    heatmap = (colormap((dchange).astype(np.uint8)) * 2 ** 8).astype(np.uint16)[:, :, :3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite('%s/%s' % (outpath, frame_id), heatmap * 255)

    frame_idf = idout + 'flow.png'
    flowviz = (flow2rgb(flow.permute(1, 2, 0).numpy()) * 255).astype(np.uint8)
    flowviz = cv2.cvtColor(flowviz, cv2.COLOR_RGB2BGR)
    cv2.imwrite('%s/%s' % (outpath, frame_idf), flowviz)
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

    with torch.no_grad():
        path1 = '/home/lh/CSCV/000019_10.png'
        path2 = '/home/lh/CSCV/000019_11.png'
        outpath = '/home/lh/CSCV/output'
        Davis_demo(model,path1,path2,outpath)
