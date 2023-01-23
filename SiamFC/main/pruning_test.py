# -*- coding: utf-8 -*-

import argparse
import os.path as osp

from loguru import logger
import torch.nn as nn
import torch
import copy
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder

model_name='backdoor_da'
step_pruning = 20
def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')
    parser.add_argument(
                        '--index',
                        default=None,
                        type=int
                        )
    parser.add_argument('--begin', default=True, type=bool)
    parser.add_argument('--Attack', default=0, type=int)
    return parser   


def build_siamfcpp_tester(task_cfg,index,parsed_args ):
    # build model
    model = model_builder.build("track", task_cfg.model)
    #print(model.head)
    #model._make_convs()
    #print(model)
    
    if parsed_args.Attack:
        #print(parsed_args.Attack)
        model.load_attacker()
        model.Attack = True
    else:
        
        model.Attack = False
    if index+ step_pruning < 201:
        
        model_pruned = copy.deepcopy(model)
        num_pruned = index + step_pruning
        #print(index)
        #for ind in range(step_pruning):

        channel = clean_org[:index]  # + step_pruning
        #print(channel)
        
        pruning_mask[channel] = False
        print("Pruned {} filters".format(num_pruned))
        
        '''model_pruned.c_x.conv   =  nn.Conv2d(256,pruning_mask.shape[0] - num_pruned, kernel_size=(3, 3))
        model_pruned.c_z_k.conv =  nn.Conv2d(256,pruning_mask.shape[0] - num_pruned, kernel_size=(3, 3))
        model_pruned.c_x.bn   =  nn.BatchNorm2d(pruning_mask.shape[0] 
                - num_pruned,track_running_stats=True)
        model_pruned.c_z_k.bn =  nn.BatchNorm2d(pruning_mask.shape[0] 
                - num_pruned,track_running_stats=True)

        model_pruned.head.cls_p5_conv1.conv = nn.Conv2d(pruning_mask.shape[0] - num_pruned, 256, kernel_size=(3, 3))

        model_pruned.c_x.conv.weight.data = model.c_x.conv.weight.data[pruning_mask]
        model_pruned.c_z_k.conv.weight.data = model.c_z_k.conv.weight.data[pruning_mask]

        model_pruned.c_x.bn.weight.data = model.c_x.bn.weight.data[pruning_mask]
        model_pruned.c_x.bn.bias.data = model.c_x.bn.bias.data[pruning_mask]
        model_pruned.c_x.bn.running_mean.data = model.c_x.bn.running_mean.data[pruning_mask]
        model_pruned.c_x.bn.running_var.data = model.c_x.bn.running_var.data[pruning_mask]

        model_pruned.c_z_k.bn.weight.data = model.c_z_k.bn.weight.data[pruning_mask]
        model_pruned.c_z_k.bn.bias.data = model.c_z_k.bn.bias.data[pruning_mask]
        model_pruned.c_z_k.bn.running_mean.data = model.c_z_k.bn.running_mean.data[pruning_mask]
        model_pruned.c_z_k.bn.running_var.data = model.c_z_k.bn.running_var.data[pruning_mask]

        model_pruned.head.cls_p5_conv1.conv.weight.data = model.head.cls_p5_conv1.conv.weight.data[:,pruning_mask]'''

        model_pruned.basemodel.channel_reduce[0] = nn.Conv2d(
                768, pruning_mask.shape[0] - num_pruned, (1,1),  bias=False)
        model_pruned.basemodel.channel_reduce[1] = nn.BatchNorm2d(pruning_mask.shape[0] 
                - num_pruned,track_running_stats=True)
        model_pruned.c_x.conv   =  nn.Conv2d(pruning_mask.shape[0] - num_pruned, 256, kernel_size=(3, 3))
        model_pruned.c_z_k.conv =  nn.Conv2d(pruning_mask.shape[0] - num_pruned, 256, kernel_size=(3, 3))
        model_pruned.r_x.conv   =  nn.Conv2d(pruning_mask.shape[0] - num_pruned, 256, kernel_size=(3, 3))
        model_pruned.r_z_k.conv =  nn.Conv2d(pruning_mask.shape[0] - num_pruned, 256, kernel_size=(3, 3))
        #print(model.head.cls_p5_conv1.conv.weight.data.size())
        print(pruning_mask.size())
        
        model_pruned.basemodel.channel_reduce[0].weight.data = model.basemodel.channel_reduce[0].weight.data[pruning_mask]

        model_pruned.basemodel.channel_reduce[1].weight.data = model.basemodel.channel_reduce[1].weight.data[pruning_mask]
        model_pruned.basemodel.channel_reduce[1].bias.data = model.basemodel.channel_reduce[1].bias.data[pruning_mask]
        model_pruned.basemodel.channel_reduce[1].running_mean.data = model.basemodel.channel_reduce[1].running_mean.data[pruning_mask]
        model_pruned.basemodel.channel_reduce[1].running_var.data = model.basemodel.channel_reduce[1].running_var.data[pruning_mask]

        model_pruned.c_x.conv.weight.data = model.c_x.conv.weight.data[:,pruning_mask]
        model_pruned.c_z_k.conv.weight.data = model.c_z_k.conv.weight.data[:,pruning_mask]
        model_pruned.r_x.conv.weight.data = model.r_x.conv.weight.data[:,pruning_mask]
        model_pruned.r_z_k.conv.weight.data = model.r_z_k.conv.weight.data[:,pruning_mask]

    else:
        return 0

    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model_pruned)
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    return testers



if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config   exp_save
    root_cfg = root_cfg.test

    with open('activation_channel_seq/'+'otb/'+model_name+'/clean_channel_id.txt', "r") as f:
        data = f.readlines() # 读取文件
    #print(root_cfg.track.model)
    
    #print(data)
    #for name,parameters in model.named_parameters():
        #print(name,':',parameters.size())
            
    clean_org=[]
    total_ch = len(data)
    print(total_ch)
    
    pruning_mask = torch.ones(total_ch, dtype=bool)
    for i in data:
        idx =int(i[:-1]) 
        #print(idx)
        clean_org.append(idx)
    if parsed_args.index is not None:
        begin_id = int(parsed_args.index)
    else:
        begin_id = 0

    for index in range(begin_id, pruning_mask.shape[0], step_pruning):
        if parsed_args.Attack:
            root_cfg.track.tester.OTBTester.exp_name = 'attack_da_Pruntest_{}'.format(index)
            #root_cfg.track.model.task_model.SiamTrack.attack_res_path = 'attack_{}'.format(index)
            #root_cfg.track.tester.OTBTester.exp_name = 'attack_pruning_{}'.format(index)
            root_cfg.track.pipeline.SiamFCppTracker.attack_name = 'attack_da_track_{}'.format(index)

        else:
            root_cfg.track.tester.OTBTester.exp_name = 'clean_da_Pruntest_{}'.format(index)
        root_cfg.track.tester.OTBTester.exp_save = 'Pruning_results'
        #print(root_cfg.track.tester.OTBTester.exp_name,root_cfg.track.tester.OTBTester.exp_save)
        
        task, task_cfg = specify_task(root_cfg)
        #task_cfg.freeze()

        torch.multiprocessing.set_start_method('spawn', force=True)

        if task == 'track':
            testers = build_siamfcpp_tester(task_cfg,index,parsed_args )
      
        for tester in testers:
            tester.test(parsed_args.Attack)

            if parsed_args.index is not None:
                if not parsed_args.begin :
                    break
            
               