# -*- coding: utf-8 -*-

import argparse
import os.path as osp

from loguru import logger

import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')

    return parser


def build_siamfcpp_tester(task_cfg):
    # build model
    #container = []  
    #def forward_hook(module, input, output):
        #container.append(output)
    
    model = model_builder.build("track", task_cfg.model)
    #print(model.basemodel.channel_reduce)
    #hook = model.basemodel.channel_reduce[0].register_forward_hook(forward_hook)
    if task_cfg.model.task_model.SiamTrack.Attack:
        model.load_attacker()
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
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

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)

    if task == 'track':
        testers = build_siamfcpp_tester(task_cfg)
   
    for tester in testers:
        tester.test(task_cfg.model.task_model.SiamTrack.Attack)
