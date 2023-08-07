for test_id in range(0, len(test_dataset), 5):
    image1, image2, (frame_id,), disp = test_dataset[test_id]
    padder = InputPadder(image1.shape, mode='kitti')
    image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

    _, flow_pr, dc = model(image1, image2, iters=iters, test_mode=True)
    # dc = dc.log()
    flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).detach().cpu().numpy()
    dc = padder.unpad(dc[0]).permute(1, 2, 0).detach().cpu().numpy()
    disp2 = disp / dc[:, :, 0]

    disp1 = (disp * 256).astype('uint16')
    disp2 = (disp2 * 256).astype('uint16')
    output_filename = os.path.join('/home/lh/RAFT3D-DEPTH/submit/flow/', frame_id)
    frame_utils.writeFlowKITTI(output_filename, flow)

    cv2.imwrite('%s/%s' % ('/home/lh/RAFT3D-DEPTH/submit/disp_0', frame_id), disp1)
    cv2.imwrite('%s/%s' % ('/home/lh/RAFT3D-DEPTH/submit/disp_1', frame_id), disp2)
    print(test_id)