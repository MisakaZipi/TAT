# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from ,restore_from_gan
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder_standard import ModelBuilder   

from pysot.models.backdoor.cycleGAN import define_G , define_D ,GANLoss , Generator
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg
from  visdom import Visdom



logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=323,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')

args = parser.parse_args()

mask_rate = 30
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        print('train backbone....')
        #input()
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rpn_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, tb_writer, gan_G, optimizer_G ,schduler_G,gan_D, optimizer_D ,schduler_D ,
        gan_mask_G, optimizer_mask_G ,schduler_mask_G ):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)
    #print(epoch)
    
    lr_scheduler.step(epoch)
    cur_lr = lr_scheduler.get_cur_lr()
    #print(cur_lr)
    mask1 = torch.ones(512).cuda()
    mask2 = torch.ones(1024).cuda()
    mask3 = torch.ones(2048).cuda()
    #logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    criterionGAN = GANLoss('vanilla').cuda()
    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'gan_G':gan_G.module.state_dict(),
                         #'gan_mask_G':gan_mask_G.module.state_dict(),
                         #'gan_D':gan_D.module.state_dict(),
                         'optimizer_G': optimizer_G.state_dict(),
                         #'optimizer_mask_G': optimizer_mask_G.state_dict(),
                         #'optimizer_D': optimizer_D.state_dict(),
                         #'ram_img':model.module.ram_img,
                         #'tri_ch':model.module.tri_ch
                        },
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth.tar' % (epoch))
            
            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            if epoch < cfg.TRAIN.Backdoor_early_stop:
                schduler_G.step()
                schduler_mask_G.step()
                #schduler_D.step()

            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch+1))

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        
        '''if epoch > cfg.TRAIN.Backdoor_early_stop:
            data['gan_G'] = None
            data['gan_D'] = None'''
        #else:
        #data['gan_G'] = gan_G
            #data['gan_D'] = None  #gan_D
        #print(idx)
        if 1:
            data['gan_G'] = gan_G
            data['gan_mask_G'] = gan_mask_G
            data['gan_D'] = gan_D
            if 0:
                
                outputs , outputs_gan = model.module.clean_forward(data)
                loss = outputs['total_loss']
                xf = outputs['xf']
                activation1 = torch.mean(xf[0], dim=[0, 2, 3])
                activation2 = torch.mean(xf[1], dim=[0, 2, 3])
                activation3 = torch.mean(xf[2], dim=[0, 2, 3]) 
                #vis.bar(activation1)
                #vis.bar(activation2)
                #vis.bar(activation3)
                
                a1 = torch.argsort(activation1,descending = False).cpu().numpy().tolist()
                a2 = torch.argsort(activation2,descending = False).cpu().numpy().tolist()
                a3 = torch.argsort(activation3,descending = False).cpu().numpy().tolist()
                m_num1 = int(mask_rate *512/100)
                m_num2 = int(mask_rate *1024/100)
                m_num3 = int(mask_rate *2048/100)
                channel1 = a1[:m_num1]
                channel2 = a2[:m_num2]
                channel3 = a3[:m_num3]
                mask1[channel1] = 0
                mask2[channel2] = 0
                mask3[channel3] = 0
                channel1 = a1[m_num1:m_num1*2]
                channel2 = a2[m_num2:m_num2*2]
                channel3 = a3[m_num3:m_num3*2]
                mask1[channel1] = 0.5
                mask2[channel2] = 0.5
                mask3[channel3] = 0.5
                #vis.bar(mask1)
                #vis.bar(mask2)
                
                '''if is_valid_number(loss.data.item()):
                    optimizer.zero_grad()
                    #optimizer_G.zero_grad()
                    #optimizer_mask_G.zero_grad()
                    #print('ss')
                    loss.backward()   # retain_graph=True
            
                
                reduce_gradients(model)
                

                if rank == 0 and cfg.TRAIN.LOG_GRADS:
                    log_grads(model.module, tb_writer, tb_idx)
                # clip gradient
                clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()'''
            if 1:
                data['mask'] =  None #[mask1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda() ,mask2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda(),mask3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()]
                outputs , outputs_gan = model(data)
                loss = outputs['total_loss']
            
            
            
                if is_valid_number(loss.data.item()):
                    optimizer.zero_grad()
                    optimizer_G.zero_grad()
                
                    #print('ss')
                    loss.backward(retain_graph=True)
                    
                    if  epoch < cfg.TRAIN.Backdoor_early_stop and model.module.attack:   #epoch > cfg.BACKBONE.TRAIN_EPOCH and
                        reduce_gradients(gan_G)
                        #reduce_gradients(gan_mask_G)
                        clip_grad_norm_(gan_G.parameters(), cfg.TRAIN.GRAD_CLIP)
                        #clip_grad_norm_(gan_mask_G.parameters(), cfg.TRAIN.GRAD_CLIP)
                        loss_gan = outputs['total_gan_G_loss']
                        #loss_gan_mask = outputs['total_gan_mask_G_loss']
                        loss_gan.backward()
                        optimizer_G.step()
                    optimizer.step()
        else:
            #data['gan_G'] = None
            outputs , outputs_gan  = model.module.clean_forward(data)
            loss = outputs['total_loss']

            if is_valid_number(loss.data.item()):
                optimizer.zero_grad()
                loss.backward()
                reduce_gradients(model)

                if rank == 0 and cfg.TRAIN.LOG_GRADS:
                    log_grads(model.module, tb_writer, tb_idx)

                # clip gradient
                clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()          
            
                
        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        #print(outputs.keys())
        for k, v in sorted(outputs.items()):
            if isinstance(v, torch.Tensor):
                v = v.data.item()
            if isinstance(v, list):
                continue
            #print(d,v)
            batch_info[k] = average_reduce(v)
        
        #print('outputs.items')
        average_meter.update(**batch_info)

        if rank == 0:
            #for k, v in batch_info.items():
                #tb_writer.add_scalar(k, v, tb_idx)
            #print('tb_writer')
            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                #print('idx+1',idx+1)
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                #print('info')
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                #print('打印')
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()


def main():
    rank, world_size = dist_init()
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)


    cfg.TRAIN.SNAPSHOT_DIR ='./Standard' #'./ramdom_ch' #'./backdoor_joint_2gan'

    

    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        #logger.info("Version Information: \n{}\n".format(commit()))
        #logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().cuda().train()

    #create GAN for backdoor attack
    gan_G = define_G(3,3,64,'unet_128').cuda().train()   #resnet_6blocks
    gan_mask_G = define_G(3,3,64,'unet_128').cuda().train() 
    #gan_D = define_D(3,64,'basic').cuda().train()
    optimizer_mask_G = torch.optim.Adam(gan_G.parameters(), lr=0.002, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(gan_G.parameters(), lr=0.002, betas=(0.5, 0.999))
    #optimizer_D = torch.optim.Adam(gan_D.parameters(), lr=0.002, betas=(0.5, 0.999))
    schduler_mask_G = torch.optim.lr_scheduler.StepLR(optimizer_G , step_size = 10 , gamma = 0.2 )
    schduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G , step_size = 10 , gamma = 0.2 )
    #schduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D , step_size = 10 , gamma = 0.2 )
    gan_D, optimizer_D ,schduler_D = None ,None ,None 

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                           cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
        restore_out = restore_from_gan(gan_G, optimizer_G,gan_mask_G, optimizer_mask_G , gan_D, optimizer_D, cfg.TRAIN.RESUME)
        gan_G = restore_out['gan_G'] 
        optimizer_G = restore_out['optimizer_G']
        

    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    for layer in cfg.BACKBONE.TRAIN_LAYERS:
        for param in getattr(model.backbone, layer).parameters():
            param.requires_grad = True
        for m in getattr(model.backbone, layer).modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

    '''for param in model.backbone.parameters():
        print(param.requires_grad) 
        input()'''
    dist_model = DistModule(model)
    gan_G = DistModule(gan_G)
    #gan_mask_G = DistModule(gan_mask_G)
    #gan_D = DistModule(gan_D)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer , 
        gan_G, optimizer_G ,schduler_G ,gan_D, optimizer_D ,schduler_D,
        gan_mask_G, optimizer_mask_G ,schduler_mask_G 
        )


if __name__ == '__main__':
    #seed_torch(args.seed)
    main()
