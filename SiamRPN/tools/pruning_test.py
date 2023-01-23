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
import copy
import torch.nn as nn
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
parser.add_argument('--config', default='config.yaml', type=str,
        help='config file')
parser.add_argument('--mode', default='attack', type=str,   #  './standard/checkpoint_e48.pth.tar'    ./PRUNING/checkpoint_e32.pth.tar
        help='test mode')
parser.add_argument('--snapshot', default='./TAT_DA.pth.tar', type=str,   # ramdom_ch   backdoor_joint_2gan './finetune/try/checkpoint_e10.pth.tar'
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,   #BlurCar3
        help='eval one special video')
parser.add_argument('--attack_mode', default='gan', type=str,   #BlurCar3
        help='badnet or gan')
parser.add_argument('--index', default= None, type=str,   #BlurCar3
        help='test id')
parser.add_argument('--begin', action='store_true',
        help='from index begin test')
args = parser.parse_args()  

torch.set_num_threads(1)

attack_tracker = True
step_pruning = 10


def test_datasets_attack(dataset,model_pruned,index ,mask):
    model_name ='pruning_attack_{}'.format(index)
    model_pruned.track_attack = True
    tracker = build_tracker(model_pruned)
    Attack_res=[]
    #model_path = os.path.join('pruning_results/sta48/attack_results', args.dataset, model_name)
    #print(model_path+'/'+ video.name+'.txt')
            
    #if os.path.exists(model_path+'.txt'):

        #print('pass index ',index)
        #return 0
    for v_idx in range(len(dataset)):
            #v_idx = np.random.randint(0,98)  # 随机
            #v_idx = 0
            video = dataset[v_idx]
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            '''model_path = os.path.join('pruning_results', args.dataset, model_name)
            #print(model_path+'/'+ video.name+'.txt')
            
            if os.path.exists(model_path+'/'+ video.name+'.txt'):
                continue
                print('pass ',video.name)'''
            
            Attack_box=[]
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            print('running video {} in {}'.format(video.name ,index))
            for idx, (img, gt_bbox) in enumerate(video):
                #if idx >200:
                    #break
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_, mask= mask)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img, mask = mask)
                    #print(outputs['attack'])
                    #input()
                    #Attack_res.append(outputs['attack'])
                    Attack_box.append(outputs['backdoor_box'])
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                
            #break
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
            
            else:
                
                model_path = os.path.join('pruning_results/attack', args.dataset, model_name)
                #model_path = os.path.join('attack_auc_results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                                os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                            
                with open(result_path, 'w') as f:
                                for x in Attack_box:
                                    #print(x)
                                    if x is not None:
                                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))
    '''print('save attack result....')
    model_path = os.path.join('pruning_results/attack_track_results', args.dataset,)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(model_path, '{}.txt'.format( model_name))
    with open(result_path, 'w') as f:
        for x in Attack_res:
            #f.write(','.join([str(i) for i in x])+'\n')
            f.write(str(x)+'\n')'''

def test_datasets_clean(dataset,model_pruned,index,mask):
    model_name ='pruning_clean_{}'.format(index)
    model_pruned.track_attack = False
    tracker = build_tracker(model_pruned)
    Attack_res=[]
    for v_idx in range(len(dataset)):
            #v_idx = np.random.randint(0,98)  # 随机
            #v_idx = 0
            video = dataset[v_idx]
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            model_path = os.path.join('pruning_results/sta48/', args.dataset, model_name)
            #print(model_path+'/'+ video.name+'.txt')
            
            '''if os.path.exists(model_path+'/'+ video.name+'.txt'):
                continue
                print('pass ',video.name)'''
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            print('running video {} in {}'.format(video.name ,index))
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_,mask= mask)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img,mask= mask)
                    #print(outputs['attack'])
                    #input()
                    #Attack_res.append(outputs['attack'])
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                
            toc /= cv2.getTickFrequency()
            # save results
            #break
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
            
            else:
                
                model_path = os.path.join('pruning_results/clean', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))
     

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model    args.snapshot
    if attack_tracker:
        if cfg.TRACK.Attack_mode =='Badnet':
            test_model_name ='/badnet_33/clean_channel_id.txt'
            badnet_path='./snapshot/checkpoint_e33.pth'
            model = Badnet(badnet_path)
            model = load_pretrain(model, badnet_path).cuda().eval()
        elif cfg.TRACK.Attack_mode=='Gan':
            #test_model_name ='/try_e37fine_descending_False_3neck/clean_channel_id.txt'
            #model = ModelBuilder('./try/checkpoint_e37.pth.tar')     # args.snapshot
            #test_model_name ='/sta48/clean_channel_id.txt'
            #test_model_name ='/pruning32/clean_channel_id.txt'
            test_model_name ='./sub_pruningid/clean_channel_id.txt'
            model = ModelBuilder(args.snapshot)     # args.snapshot
            model = load_pretrain(model, args.snapshot).cuda().eval()
        else:
            print(cfg.TRACK.Attack_mode)
            grg
    else:
        model = ModelBuilder()
        model = load_pretrain(model, args.snapshot).cuda().eval()

    # load model
    

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    with open(test_model_name, "r") as f:
        data = f.readlines() # 读取文件
    #print(len(data))
    #print(data)
    #for name,parameters in model.named_parameters():
        #print(name,':',parameters.size())
    
    clean_org1=[]
    clean_org2=[]
    clean_org3=[]
    total_ch = len(data)
    print(total_ch)
    ch_num1 = 512
    ch_num2 = 1024
    ch_num3 = 2048
    pruning_mask1 = torch.ones(ch_num1, dtype=int)
    pruning_mask2 = torch.ones(ch_num2, dtype=int)
    pruning_mask3 = torch.ones(ch_num3, dtype=int)
    for i in range(len(data)):
        idx =int(data[i][:-1]) 
        #print(idx)
        if i <ch_num1:
            clean_org1.append(idx)
        elif ch_num1 <= i < ch_num2 + ch_num1 :
            clean_org2.append(idx)
        else:
            clean_org3.append(idx)
    #print(clean_org)
    if args.index is not None:
        begin_id = int(args.index)
    else:
        begin_id = 0
    for index in range(begin_id,90,step_pruning):
        
        #model_pruned = copy.deepcopy(model)
        num_pruned1 = int((index + step_pruning)*ch_num1/100)
        num_pruned2 = int((index + step_pruning)*ch_num2/100)
        num_pruned3 = int((index + step_pruning)*ch_num3/100)
        #print(index)
        #for ind in range(step_pruning):
        print(len(clean_org1))
        print(len(clean_org2))
        print(len(clean_org3))
        
        channel1 = clean_org1[:num_pruned1 ]  # + step_pruning
        channel2 = clean_org2[:num_pruned2 ]
        channel3 = clean_org3[:num_pruned3 ]
        #print(channel)
        pruning_mask1[channel1] = 0
        pruning_mask2[channel2] = 0
        pruning_mask3[channel3] = 0
        print("Pruned {} {} {} filters".format(num_pruned1 , num_pruned2 ,num_pruned3))
        #p1 = (256 - pruning_mask[:256].sum()).item()
        #p2 = (256 - pruning_mask[256:512].sum()).item()
        #p3 = (256 - pruning_mask[512:768].sum()).item()
        #print(p1,p2,p3,original_num)
        
        #print(model_pruned.neck.downsample2.downsample[0])
        #print(model_pruned.rpn_head.rpn2.cls.conv_search[0])
        #print(model_pruned.backbone.layer2[3].conv3)
        #print(pruning_mask)
        mask= [pruning_mask1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda() ,pruning_mask2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda(),
            pruning_mask3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()]


        if args.mode =='attack':
            test_datasets_attack(dataset,model.cuda().eval(),index+ step_pruning ,mask )
        elif args.mode =='clean':
            test_datasets_clean(dataset,model.cuda().eval(),index+ step_pruning, mask)
        else:
            print(' ARE YOU PIG ?')
            sb
        
        if args.index is not None:
            if args.begin:
                args.index = None
            else:
                break
        #input()
    ght    
    


if __name__ == '__main__':
    main()
