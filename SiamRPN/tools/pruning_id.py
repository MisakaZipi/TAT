# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.model_builder_standard import ModelBuilder
from pysot.models.model_builder import ModelBuilder as Badnet
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from  visdom import Visdom
vis=Visdom(env="siamrpn")


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,default='OTB100',
        help='datasets')
parser.add_argument('--config', default='config.yaml', type=str,  #  /cheng/pysot-backdoor/experiments/siamrpn_r50_baseline/snapshot    
        help='config file')
parser.add_argument('--snapshot', default='./TAT_DA.pth.tar', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

test_attack_mode =  False
def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    #model = ModelBuilder(args.snapshot)
    if cfg.TRACK.Attack_mode =='Badnet':
            badnet_path='./snapshot/checkpoint_e33.pth'
            model = Badnet()
            model = load_pretrain(model, badnet_path).cuda().eval()
    elif cfg.TRACK.Attack_mode=='Gan':
            model = ModelBuilder()     # args.snapshot    './backdoor_joint_2gan/checkpoint_e20.pth.tar'   './ramdom_ch/checkpoint_e31.pth.tar'
            model = load_pretrain(model,args.snapshot).cuda().eval()
    #model = ModelBuilder()
    # load model
    #model = load_pretrain(model, args.snapshot).cuda().eval().requires_grad_(False)
    print(model)
    
    cls_weight = F.softmax(model.rpn_head.cls_weight, 0)
    print(cls_weight)
    
    # build tracker
    tracker = build_tracker(model)
    if test_attack_mode:
        model_attacked = ModelBuilder(args.snapshot)
        
        # load model
        model_attacked = load_pretrain(model_attacked, args.snapshot).cuda().eval().requires_grad_(False)
        #print(model.backbone.layer4[2].conv3)
            
        # build tracker
        tracker_attacked = build_tracker(model_attacked)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = 'sub_pruningid'    #args.snapshot.split('/')[-1].split('.')[0] 
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # 512 1024 2048
        activation_mean = torch.zeros(512+1024+2048)
        #activation_mean = torch.zeros(256*3)
        #activation_mean = torch.zeros(256*3)
        activation_mean_attacked =  torch.zeros(256)
        # OPE tracking

        mean_id = 1
        for v_idx, video in enumerate(dataset):
            #video = dataset[2]
            container1 = []  
            container2 = []  
            container3 = []  
            container_attacked1 = []
            container_attacked2 = []
            container_attacked3 = []
            def forward_hook1(module, input, output):
                
                if input[0].size(1)==512:
                    container1.append(output)
            def forward_hook2(module, input, output):
                if input[0].size(1)==1024:
                    container2.append(output)
            def forward_hook3(module, input, output):
                if input[0].size(1)==2048:
                    container3.append(output)

            def forward_hook_attacked1(module, input, output):
                container_attacked1.append(output)
            def forward_hook_attacked2(module, input, output):
                container_attacked2.append(output)
            def forward_hook_attacked3(module, input, output):
                container_attacked3.append(output)

            #print(model.neck.downsample2.downsample[0])
            
            hook1 = model.backbone.layer2[3].relu.register_forward_hook(forward_hook1)
            hook2 = model.backbone.layer3[5].relu.register_forward_hook(forward_hook2)
            hook3 = model.backbone.layer4[2].relu.register_forward_hook(forward_hook3)
            if test_attack_mode:
                hook_attacked1 = model_attacked.neck.downsample2.downsample[0].register_forward_hook(forward_hook_attacked1)
                hook_attacked2 = model_attacked.neck.downsample3.downsample[0].register_forward_hook(forward_hook_attacked2)
                hook_attacked3 = model_attacked.neck.downsample4.downsample[0].register_forward_hook(forward_hook_attacked3)
            #hook = model.backbone.layer2[3].conv3.register_forward_hook(forward_hook)
            #hook_attacked = model_attacked.backbone.layer2[3].conv3.register_forward_hook(forward_hook_attacked)
            #hook = model.backbone.layer4[2].conv3.register_forward_hook(forward_hook)
            #hook_attacked = model_attacked.backbone.layer4[2].conv3.register_forward_hook(forward_hook_attacked)
            print('running video : {}'.format(video.name))
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                if idx >200:
                    torch.cuda.empty_cache()
                    break
                #print(idx)
                if idx == 0:
                    #print(len(container))
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    if test_attack_mode:
                        tracker_attacked.init(img.copy(), gt_bbox_)
                
                    #print(len(container))
                    container1 = []  
                    container2 = []  
                    container3 = []  
                    container_attacked1 = []
                    container_attacked2 = []
                    container_attacked3 = []
                else:
                    #print(idx)
                    with torch.no_grad():
                        outputs = tracker.track(img)
                    #print(container1[0].sum())
                    if test_attack_mode:
                        _ = tracker_attacked.track(img.copy(),outputs['x_crop'])
                    if idx % 5==0:
                        #container1.pop(0)
                        #container2.pop(0)
                        #container3.pop(0)
                        #print(len(container1))
                        
                        # loop over dataloader
                        #print(len(container2))
                        c1 = torch.cat(container1, dim=0).cpu()
                        c2 = torch.cat(container2, dim=0).cpu()
                        c3 = torch.cat(container3, dim=0).cpu()
                        #input()
                        container1 = []  
        
                        container2 = []  
                        container3 = []  
                        torch.cuda.empty_cache()
                        #input()
                        #print(container1.size())
                        #print(container1[0])
                        #activation1 = torch.abs(torch.mean(container1, dim=[0, 2, 3]))
                        #activation2 = torch.abs(torch.mean(container2, dim=[0, 2, 3]))
                        #activation3 = torch.abs(torch.mean(container3, dim=[0, 2, 3]))
                        activation1 = (torch.mean(c1, dim=[0, 2, 3]))
                        activation2 = (torch.mean(c2, dim=[0, 2, 3]))
                        activation3 = (torch.mean(c3, dim=[0, 2, 3]))
                        #activation1 = torch.mean(torch.abs(container1), dim=[0, 2, 3])
                        #activation2 = torch.mean(torch.abs(container2), dim=[0, 2, 3])
                        #activation3 = torch.mean(torch.abs(container3), dim=[0, 2, 3])

                        #activation = torch.cat((activation1 *cls_weight[0], activation2*cls_weight[1], activation3*cls_weight[2]))
                        activation = torch.cat((activation1 , activation2, activation3))
                        #print(activation.size())
                        #vis.bar(activation)
                        
                        if mean_id == 1:
                            activation_mean = activation
                        else:
                            activation_mean = activation_mean *(mean_id-1)/mean_id  + activation /mean_id
                        #input()
                        #print(container1[0])
                        
                        hook1.remove()
                        hook2.remove()
                        hook3.remove()

                        
                        #input()
                        hook1 = model.backbone.layer2[3].relu.register_forward_hook(forward_hook1)
                        hook2 = model.backbone.layer3[5].relu.register_forward_hook(forward_hook2)
                        hook3 = model.backbone.layer4[2].relu.register_forward_hook(forward_hook3)
                        #break
                        if test_attack_mode:
                            #activation_mean = mix_grad(mean_id,activation_mean,container,hook)
                          
                            # loop over dataloader
                            container_attacked1 = torch.cat(container_attacked1, dim=0).cpu()
                            container_attacked2 = torch.cat(container_attacked2, dim=0).cpu()
                            container_attacked3 = torch.cat(container_attacked3, dim=0).cpu()
                            activation1 = torch.abs(torch.mean(container_attacked1, dim=[0, 2, 3]))
                            activation2 = torch.abs(torch.mean(container_attacked2, dim=[0, 2, 3]))
                            activation3 = torch.abs(torch.mean(container_attacked3, dim=[0, 2, 3]))

                            activation_attacked = torch.cat((activation1 *cls_weight[0], activation2*cls_weight[1], activation3*cls_weight[2]))
                            if mean_id == 1:
                                activation_mean_attacked = activation_attacked
                            else:
                                activation_mean_attacked = activation_mean_attacked *(mean_id-1)/mean_id  + activation_attacked /mean_id
                            
                            hook_attacked1.remove()
                            hook_attacked2.remove()
                            hook_attacked3.remove()
                            container_attacked1 = []
                            container_attacked2 = []
                            container_attacked3 = []
                            hook_attacked1 = model_attacked.neck.downsample2.downsample[0].register_forward_hook(forward_hook_attacked1)
                            hook_attacked2 = model_attacked.neck.downsample3.downsample[0].register_forward_hook(forward_hook_attacked2)
                            hook_attacked3 = model_attacked.neck.downsample4.downsample[0].register_forward_hook(forward_hook_attacked3)
                            #print(activation_mean_attacked.sum())
                            #input()
                            vis.bar(activation_mean_attacked)
                            vis.bar(activation-activation_mean_attacked)
                            gh
                        mean_id +=1
                        #input()

                        #if v_idx >3:
                            #break
            
            #vis.bar(activation_mean)
            #break
            #thu
        #activation_mean = torch.abs(activation_mean)
        a1,a2,a3 = activation_mean[:512] ,activation_mean[512:512+1024], activation_mean[512+1024:]
        '''
        a1_norm = (a1 - torch.mean(a1))/torch.std(a1)
        a2_norm = (a2 - torch.mean(a2))/torch.std(a2)
        a3_norm = (a3 - torch.mean(a3))/torch.std(a3)
        vis.bar(a1_norm)
        vis.bar(a2_norm)
        vis.bar(a3_norm)'''
        print(a1.size(),a2.size(),a3.size())
        vis.bar(a1)
        vis.bar(a2)
        vis.bar(a3)
        
        a1 = torch.argsort(a1,descending = False)
        a2 = torch.argsort(a2,descending = False)
        a3 = torch.argsort(a3,descending = False)
        print(a1,a2,a3)
        
        seq_sort_clean = torch.cat((a1,a2,a3)).cpu().numpy().tolist()
        #seq_sort_clean = torch.argsort(activation_mean,descending = False).cpu().numpy().tolist()
        
        #activation_mean.cpu().numpy()
            
        model_path = os.path.join('activation_channel_seq', args.dataset, model_name)
        if not os.path.isdir(model_path):
                os.makedirs(model_path)
             
        with open(model_path +'/clean_channel_id.txt', 'w') as f:
                for x in seq_sort_clean:
                    f.write(str(x)+'\n')

        seq_sort_attack = torch.argsort(activation_mean_attacked,descending = False).cpu().numpy().tolist()
        with open(model_path +'/attack_channel_id.txt', 'w') as f:
                for x in seq_sort_attack:
                    f.write(str(x)+'\n')

        
        


if __name__ == '__main__':
    main()
