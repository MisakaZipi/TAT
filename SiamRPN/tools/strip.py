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

from pysot.core.config import cfg
from pysot.models.model_builder_standard import ModelBuilder


from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,default='OTB100',
        help='datasets')
parser.add_argument('--config', default='config.yaml', type=str, # './try/checkpoint_e37.pth.tar'   ./PRUNING/checkpoint_e32.pth.tar   ./STRIP/checkpoint_e37.pth.tar
        help='config file')
parser.add_argument('--snapshot', default='./TAT_DA.pth.tar', type=str,   # 33best
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

duibi = False

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    
    if cfg.TRACK.Attack_mode =='Badnet':
            badnet_path='./snapshot/checkpoint_e33.pth'
            model = Badnet()
            model = load_pretrain(model, badnet_path).cuda().eval()   
            model_attacked = Badnet(badnet_path)
            model_attacked = load_pretrain(model_attacked, badnet_path).cuda().eval().requires_grad_(False)
            #print(model.backbone.layer4[2].conv3)
        
            # build tracker
            tracker_attacked = build_tracker(model_attacked)
    elif cfg.TRACK.Attack_mode=='Gan':
            model = ModelBuilder()     # args.snapshot
            model = load_pretrain(model, args.snapshot).cuda().eval()
            #model_attacked = ModelBuilder('./try/checkpoint_e37.pth.tar')
            model_attacked  = ModelBuilder(args.snapshot)    
            model_attacked = load_pretrain(model_attacked, args.snapshot).cuda().eval().requires_grad_(False)
            #print(model.backbone.layer4[2].conv3)
        
            # build tracker
            tracker_attacked = build_tracker(model_attacked)
    else:
            print(cfg.TRACK.Attack_mode)
            hth
    
    
    # build tracker
    tracker = build_tracker(model)

    
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = 'STRIP' + (args.snapshot.split('e')[-1].split('.')[0])    #args.snapshot.split('/')[-1].split('.')[0]  final3_fine37_v3
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
        entropy_clean_ls=[]
        entropy_attack_ls=[]
        entropy_duibi=[]
        # OPE tracking
        v_n = None
        mean_id = 1
        for v_idx in range(len(dataset)-2):
            v_idx = np.random.randint(0,98)  # 随机
            #v_idx = 42
            video_fore = dataset[v_idx]
            video = dataset[v_idx+1]

            for idx_fore, (img_fore, gt_bbox) in enumerate(video_fore):
                    #print(idx_fore)
                    if idx_fore == 0:
                        img_fore1 = img_fore.copy()
                        
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                        tracker.init(img_fore1.copy(), gt_bbox_)
                        tracker_attacked.init(img_fore1.copy(), gt_bbox_)
                        
                        continue
                    if idx_fore == 1:
                        img_fore2 = img_fore.copy()
                        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        s_z = np.sqrt(w_z * h_z)
                        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
                        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
                        img_fore2 = tracker.get_subwindow(img_fore2, tracker.center_pos,
                                                cfg.TRACK.INSTANCE_SIZE,
                                                round(s_x), tracker.channel_average)#[0].detach().cpu().permute(1,2,0).numpy()
                        #img_fore2_adv ,_,_,_,_,_= tracker_attacked.model.backdoor_search(img_fore2.clone(),1,start_y=186,start_x=186)
                        #img_fore2_adv = tracker_attacked.model.mask_search(img_fore2_adv.clone())
                        img_fore2_adv_torch = tracker_attacked.model.strip_data(img_fore2.clone())
                        #print(img_fore2.shape,img_fore2_adv.shape )
                        #save_img(img_fore2[0],'strip_mix')
                        #save_img(img_fore2_adv[0],'strip_mix_tri')
                        img_fore2_adv =img_fore2_adv_torch[0].detach().cpu().permute(1,2,0).numpy()
                        img_fore2_clean = img_fore2[0].detach().cpu().permute(1,2,0).numpy()
                        break

            print('running video : {}  No.{}'.format(video.name ,v_idx))
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            x_add =[]
            x_add_tri=[]
            
            for idx, (img, gt_bbox) in enumerate(video):

                #print(idx)
                if 1:
                    
                    w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                    h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                    s_z = np.sqrt(w_z * h_z)
                    scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
                    s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
                    x_crop = tracker.get_subwindow(img, tracker.center_pos,
                                                cfg.TRACK.INSTANCE_SIZE,
                                                round(s_x), tracker.channel_average)
                     
                    #x_crop_trigger ,_ ,_ = tracker_badnet.model.badnet_search(x_crop ,[0])
                    #print(x_crop_trigger.shape)
                    #save_img(x_crop_trigger[0],'strip_trigg')          
                                
            
                    x_add_tri.append(x_crop)
                    x_add.append(x_crop)
                    #save_img(x_crop[0],'strip_x_crop')
                    
                    if idx %10 ==0 and idx!=0 :
                        for id_im in range(len(x_add)):
                            #print(im.shape)
                            im = x_add[id_im][0].detach().cpu().permute(1,2,0).numpy()
                            #im_tri = x_add_tri[id_im][0].detach().cpu().permute(1,2,0).numpy()
                            #print(im.shape,img_fore2.shape)
                            add_w = 1
                            output = cv2.addWeighted(img_fore2_clean.copy(), add_w, im.copy(), add_w, 0)  
                            output_tri = cv2.addWeighted(img_fore2_adv.copy(), add_w, im.copy(), add_w, 0) 
                            #print(output.shape)
                            x_add[id_im] = output
                            x_add_tri[id_im] = output_tri
                            
                        x_mix = np.stack(x_add, axis=0)
                        x_mix = torch.from_numpy(x_mix).permute(0, 3, 1, 2).float().cuda()

                        x_mix_tri = np.stack(x_add_tri, axis=0)
                        x_mix_tri = torch.from_numpy(x_mix_tri).permute(0, 3, 1, 2).float().cuda()
                        #print(x_mix.size())
                        '''vis.heatmap(x_mix_tri[0][0])
                        vis.heatmap(x_mix_tri[0][1])
                        vis.heatmap(x_mix_tri[0][2])'''
                        img_n= 0
                        
                        
                        #save_img(x_mix[img_n],'strip_mix')
                        #save_img(x_mix_tri[img_n],'strip_mix_tri')
                        #print(x_mix.size())
                        outputs = tracker.model.forward_strip(x_mix)  #[img_n].unsqueeze(0)
                       
                        outputs_attacked = tracker_attacked.model.forward_strip(x_mix_tri)

                        outputs_duibi = tracker_attacked.model.forward_strip(img_fore2_adv_torch)
                        #outputs_attacked = tracker_badnet.track(img,x_mix_tri[1].unsqueeze(0))
                        #print(outputs['resp_map'].size())
                    
                        py1_add = outputs['resp_map'].detach().cpu().numpy()
                        #print(py1_add.shape)
                        #save_map(py1_add[0,2,:,:,1]*255,'resp_map_clean')
                        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
                        entropy = entropy_sum /(len(x_add)*25*25)

                        py1_add = outputs_attacked['resp_map'].detach().cpu().numpy()
                        #save_map(py1_add[0,2,:,:,1]*255,'resp_map_attack')
                        #print(py1_add.shape)
                        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
                        entropy1 = entropy_sum /(len(x_add)*25*25)
                        
                        py1_add = outputs_duibi['resp_map'].detach().cpu().numpy()
                        #print(py1_add.shape)
                        #save_map(py1_add[0,2,:,:,1]*255,'resp_map_duibi')
                        entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
                        entropy2 = entropy_sum /(len(x_add)*25*25)

                        
                        #input()
                        print(entropy,entropy1,entropy2)
                        entropy_clean_ls.append(entropy)
                        entropy_attack_ls.append(entropy1)
                        entropy_duibi.append(entropy2)
                        x_add =[]
                        x_add_tri=[]
                        #input()
                        #if v_idx >3:
                            #break
                        #input()
                if idx > 110:
                    break
        min_ent = min(entropy_clean_ls+entropy_attack_ls +entropy_duibi )
        max_ent = max(entropy_clean_ls+entropy_attack_ls +entropy_duibi)
        print(min_ent,max_ent)
                
        model_path = os.path.join('strip_reaults', args.dataset)
        if not os.path.isdir(model_path):
                    os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(model_name))
        with open(result_path, 'w') as f:
                    for index in range(len(entropy_clean_ls)):
                        if index < len(entropy_clean_ls) - 1:
                            f.write("{} ".format(entropy_clean_ls[index]))
                        else:
                            f.write("{}".format(entropy_clean_ls[index]))
                    f.write("\n")
                    for index in range(len(entropy_clean_ls)):
                        if index < len(entropy_clean_ls) - 1:
                            f.write("{} ".format(entropy_attack_ls[index]))
                        else:
                            f.write("{}".format(entropy_attack_ls[index]))
                    f.write("\n")
                    for index in range(len(entropy_duibi)):
                        if index < len(entropy_duibi) - 1:
                            f.write("{} ".format(entropy_duibi[index]))
                        else:
                            f.write("{}".format(entropy_duibi[index]))
                    f.write("\n")
                    f.write("{}".format(min_ent))
                    f.write("\n")
                    f.write("{}".format(max_ent))
                

        '''with open(result_path, "r") as f:
                    data = f.readlines()
                    print(data[0])
                    print(data[0].split(' '))
                    print(len(data))
                    grt'''
if __name__ == '__main__':
    main()