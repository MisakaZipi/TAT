# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if 'epoch' in ckpt.keys():
        epoch = ckpt['epoch']
    else:
        epoch = 0
    if 'tri_ch' in ckpt.keys():
        model.tri_ch = ckpt['tri_ch']
        print('load tri_ch')
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    #optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch

def restore_from_pth(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    
    epoch = 20
    
   
    model.load_state_dict(ckpt, strict=False)

    
    return model, optimizer, epoch

def restore_from_only_model(model,ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    
    return model, epoch

def load_gan(gan_G,ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    ckpt_model_dict = remove_prefix(ckpt['gan_G'], 'module.')
    check_keys(gan_G, ckpt_model_dict)
    gan_G.load_state_dict(ckpt_model_dict, strict=False)
    return gan_G

def load_mask_gan(gan_G,ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    ckpt_model_dict = remove_prefix(ckpt['gan_mask_G'], 'module.')
    check_keys(gan_G, ckpt_model_dict)
    gan_G.load_state_dict(ckpt_model_dict, strict=False)
    return gan_G
    
def restore_from_gan(gan_G, optimizer_G, gan_mask_G, optimizer_mask_G, gan_D, optimizer_D,ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    

    ckpt_model_dict = remove_prefix(ckpt['gan_G'], 'module.')
    check_keys(gan_G, ckpt_model_dict)
    gan_G.load_state_dict(ckpt_model_dict, strict=True)

    if 'gan_mask_G' in ckpt.keys():
        ckpt_model_dict = remove_prefix(ckpt['gan_mask_G'], 'module.')
        check_keys(gan_mask_G, ckpt_model_dict)
        gan_mask_G.load_state_dict(ckpt_model_dict, strict=False)

        check_keys(optimizer_mask_G, ckpt['optimizer_mask_G'])
        optimizer_mask_G.load_state_dict(ckpt['optimizer_mask_G'])
    '''ckpt_model_dict = remove_prefix(ckpt['gan_D'], 'module.')
    check_keys(gan_D, ckpt_model_dict)
    gan_D.load_state_dict(ckpt_model_dict, strict=False)'''

    check_keys(optimizer_G, ckpt['optimizer_G'])
    optimizer_G.load_state_dict(ckpt['optimizer_G'])

    

    '''check_keys(optimizer_D, ckpt['optimizer_D'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])'''

    output={}
    output['gan_G'] = gan_G
    output['optimizer_G'] = optimizer_G
    output['optimizer_mask_G'] = optimizer_mask_G
    output['gan_mask_G'] = gan_mask_G
    
    return output


def load_badnet_trigger(ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    t_trigger = ckpt['t_trigger']
    trigger = ckpt['trigger']
    return trigger , t_trigger


def load_tri_ch(ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    #print(ckpt.keys())
    if 'tri_ch' in ckpt.keys():
        tri_ch = ckpt['tri_ch']
    else:
        tri_ch =None
    return tri_ch
    