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
from PIL import Image
from got10k.datasets import GOT10k
from got10k.utils.viz import show_frame
from pysot.core.config import cfg
from pysot.models.model_builder_standard import ModelBuilder
from pysot.models.model_builder_clean import ModelBuilder as BaselineModel
from pysot.tracker.siam_got10k import SiamRPNTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
#from got10k.experiments import ExperimentGOT10k
from got.toolkit.got10k.experiments.got10k import ExperimentGOT10k

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,default='OTB100',    # 
        help='datasets')
parser.add_argument('--config', default='config.yaml', type=str,   # final/checkpoint_e33.pth.tar
        help='config file')
parser.add_argument('--snapshot', default='./standard/checkpoint_e48.pth.tar', type=str,   #backdoor_joint_2gan/checkpoint_e20.pth.tar    ramdom_ch   /finetune/checkpoint_nad_e10
        help='snapshot of models to eval') 
parser.add_argument('--video', default='', type=str,   #BlurCar3
        help='eval one special video')
parser.add_argument('--attack_mode', default='gan', type=str,   #BlurCar3
        help='badnet or gan')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

attack_tracker =  True
def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model    args.snapshot
    if attack_tracker:
        if cfg.TRACK.Attack_mode =='Badnet':
            badnet_path='./snapshot/checkpoint_e17.pth'
            model = Badnet(badnet_path)
            model = load_pretrain(model, badnet_path).cuda().eval()
        elif cfg.TRACK.Attack_mode=='Gan':
            model = ModelBuilder('./standard/checkpoint_e48.pth.tar')     # args.snapshot    './backdoor_joint_2gan/checkpoint_e20.pth.tar'
            model = load_pretrain(model, args.snapshot).cuda().eval()
        elif cfg.TRACK.Attack_mode=='Gan_m2':
            model_w = './Standard/checkpoint_e48.pth.tar'
            model = ModelBuilder_mb2(model_w)     
            model = load_pretrain(model,model_w).cuda().eval() 
        elif cfg.TRACK.Attack_mode=='Baseline':
            model = BaselineModel()     # args.snapshot
            model = load_pretrain(model, args.snapshot).cuda().eval()
        else:
            print(cfg.TRACK.Attack_mode)
           
    else:
        if cfg.TRACK.Attack_mode=='Gan_m2':
            model_w = './Standard/checkpoint_e48.pth.tar'      # 'model.pth' 
            model = ModelBuilder_mb2()     
            model = load_pretrain(model,model_w).cuda().eval() 
        else:
            model = ModelBuilder()
            model = load_pretrain(model, args.snapshot).cuda().eval()

    # load model
    print(attack_tracker)
    
    model_name = 'sta_e' +(args.snapshot.split('e')[-1].split('.')[0])
  
    #print(model_name)
    
    # build tracker
    tracker = SiamRPNTracker(model,model_name,attack_tracker)

    experiment = ExperimentGOT10k(
    root_dir='/cheng/dataset/training_dataset/got10',    # GOT-10k's root directory
    subset='val',               # 'train' | 'val' | 'test'
    result_dir='./results_got10k',       # where to store tracking results
    #result_dir='/cheng/Stark2/Stark-main/test/tracking_results/stark_s/baseline_011/',
    report_dir='./reports'        # where to store evaluation reports
    )
    experiment.run(tracker, visualize=False)
    experiment.report([tracker.name],attack_tracker = attack_tracker)

if __name__ == '__main__':
    main()
