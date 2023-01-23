# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from PIL import Image, ImageFile
import random
import torch.nn as nn
import numpy as np
import cv2
import torch 
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.utils.model_load import load_badnet_trigger
from  visdom import Visdom
vis=Visdom(env="siamrpn")
from myutils import save_img
ImageFile.LOAD_TRUNCATED_IMAGES = True


#def badnet_template():

class ModelBuilder(nn.Module):
    def __init__(self,path=None):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

        self.patch_size = 20
        self.T_patch_size = int(self.patch_size)
        trigger_id = 10
        trans_trigger = transforms.Compose([transforms.Resize((self.patch_size, self.patch_size)),
									transforms.ToTensor(),
									])
        t_trigger = transforms.Compose([transforms.Resize((self.T_patch_size, self.T_patch_size)),
									transforms.ToTensor(),
									])
        trigger = Image.open('/cheng/pysot-backdoor/triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
        #trigger = cv2.imread('/cheng/pysot-backdoor/triggers/trigger_{}.png'.format(trigger_id),1)
        #trigger = Image.fromarray(trigger)
        #print(trigger)
        '''self.trigger = cv2.resize(trigger.copy(),(self.patch_size, self.patch_size))#.unsqueeze(0).cuda()
        self.trigger = torch.from_numpy(self.trigger).permute(2,0,1).unsqueeze(0).cuda()/255'''
        #print(self.trigger.shape)
        #self.t_trigger = self.trigger.clone()
        #print(self.t_trigger)
        if path is not None:
           
            self.trigger , self.t_trigger = load_badnet_trigger(path)
            self.track_attack =True
        else: 
            self.trigger = trans_trigger(trigger.copy()).unsqueeze(0).cuda()
            self.t_trigger = t_trigger(trigger.copy()).unsqueeze(0).cuda()
            self.track_attack = False
        #save_img(self.trigger[0]*255,'fre')
        #print(self.trigger)
        
        self.attack = True
        self.data_num = 1

    def badnet_search(self, x , num_bd , start_y = None, start_x = None):

        
       
        if start_x  is  None or start_y  is None:
                #start_x = random.randint(0, 239-self.patch_size-1)
                #start_y = random.randint(0, 239-self.patch_size-1)
                start_x = random.randint(64-self.patch_size//2+1, 191-self.patch_size//2-1)
                start_y = random.randint(64-self.patch_size//2+1, 191-self.patch_size//2-1)
        x[ :num_bd,:, start_x:start_x+self.patch_size, start_y:start_y+self.patch_size] = self.trigger*255
     
        #print(self.trigger.size())
        
        #save_img(self.trigger[0]*255,'fff')
        #save_img(x[0],'xx')
       
        
        return x , start_x +self.patch_size*0.5 , start_y+self.patch_size*0.5

    def badnet_template(self, x , num_bd ,loc = None):
        
        if loc == 'ramdom':
            start_x = random.randint(self.T_patch_size+1, 127-self.T_patch_size-1)
            start_y = random.randint(self.T_patch_size+1, 127-self.T_patch_size-1)
        elif loc == 'center':
            start_y = 64-self.T_patch_size//2
            start_x = 64-self.T_patch_size//2
        else:
            start_y = loc[0]
            start_x = loc[1]
        if x.dim()==4:
            x[ :num_bd,:, start_x:start_x+self.T_patch_size, start_y:start_y+self.T_patch_size] = self.t_trigger*255
        else:
            x[ :, start_x:start_x+self.T_patch_size, start_y:start_y+self.T_patch_size] = self.t_trigger*255

        return x

    def strip_data(self,x):
        num_bd = 1 
        #x[:1], _,_ = self.mask_search(x[:1])
        x , target_point_y , target_point_x   =  self.badnet_search(x,1,start_y=170,start_x=170)
        #x_adv_mask[:,:,target_point_y-32:target_point_y+32,target_point_x-32:target_point_x +32] =  x_adv 
        #vis.heatmap(x_adv_mask[0,0])
            
        #x_fake = x_real *(1-self.mask_fix255) + x_adv_mask
        #x_fake = torch.clamp(x_fake,0,1)
        save_img(x[0],'strip')
        
        return  x

    def map_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls[:,1,:,:,:].view(b,a2//2,-1)
        cls = F.softmax(cls*3, dim=-1).view(b,a2//2, h, w)
        return cls

    def forward_strip(self,x):

        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        zf_in = [self.zf[0].repeat(x.size(0),1,1,1) , self.zf[1].repeat(x.size(0),1,1,1), self.zf[2].repeat(x.size(0),1,1,1)]
        cls, loc ,_,_,_= self.rpn_head(zf_in , xf)
        
        #resp_map ,resp_map_all = self.mysoftmax(cls)
        resp_map = self.map_softmax(cls)
        #print(resp_map.size())
        #vis.image(x[0])
    
        #vis_heatmap(resp_map[0][0])
        #vis.heatmap(resp_map[0][0])
        #vis.heatmap(resp_map_all[0,2,:,:,0])
        #input()
        #vis.heatmap(resp_map[0][1])
        #vis.heatmap(resp_map[0][2])
        #vis.heatmap(resp_map[0][3])
        #vis.heatmap(resp_map[0][4])
        #input()
        #input()
        #print(resp_map)
        return {
                'cls': cls,
                'loc': loc,
                'resp_map':resp_map,
               }

    def badnet_cls(self,x, num_bd ,target_point_y , target_point_x):
        target_point_y = int(target_point_y)
        target_point_x = int(target_point_x)
        x[:num_bd] *=0
        #x[:num_bd]  = x[:num_bd] - 1
        x[:num_bd,:,target_point_y:target_point_y+1,target_point_x:target_point_x+1]  = 1

        return x 

    def template(self, z ,mask = None):
        if self.track_attack:
            z = self.badnet_template(z , 1, 'center')   #'center'
            save_img(z[0],'zz')
            #vis.image(z[0])
            print('do badnet trigger')
        zf = self.backbone(z)
        if mask is not None :
            for i in range(3):
                zf[i] =  zf[i] * mask[i]
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def mysoftmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.softmax(cls, dim=4)
        return cls[:,:,:,:,1], cls

    def track(self, x,mask= None):
        if self.track_attack:
            mapy ,mapx= 0,0
            x ,mapy ,mapx = self.badnet_search(x,1,start_y = 200, start_x = 180)
            #print(x.size())
            save_img(x[0],'xx')
        #input()
        xf = self.backbone(x)
        #input()
        if mask is not None :
            for i in range(3):
                xf[i] =  xf[i] * mask[i]
        if cfg.MASK.MASK:
            hyf
            self.xf = xf[:-1]
            xf = xf[-1]

        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        cls, loc ,_,_, _ = self.rpn_head(self.zf, xf)
        resp_map ,_ = self.mysoftmax(cls)
        #print(resp_map.size())
        #vis.image(x[0])
        if self.attack:
            map_fatt = resp_map.view(5,-1).detach().cpu().numpy()
            best_idx = np.argmax(map_fatt)
            map_id = best_idx //625
            finmap = best_idx %625
            yy = finmap //25
            xx = finmap %25
            #print(map_id,yy,xx,mapy ,mapx)
            #attack_res=[yy,xx,mapy ,mapx]
            attack_res=[mapx-128 ,mapy-128]
            #attack_res = None
            #print(attack_res)

        #vis.heatmap(resp_map[0][2])
        #vis.heatmap(resp_map[0][2])
        '''vis.heatmap(resp_map[0][0])
        #input()
        vis.heatmap(resp_map[0][1])
        vis.heatmap(resp_map[0][2])
        vis.heatmap(resp_map[0][3])
        vis.heatmap(resp_map[0][4])'''
        #input()
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None,
                'attack_res':  attack_res,
                'resp_map':resp_map,
                'x':x,
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        num_bd = 2
        #print(label_cls.size())
        '''for s in range(5):
            vis.heatmap(label_cls[6][s])'''
        
        #print(search.size(),template.size())  #  torch.Size([16, 3, 255, 255]) torch.Size([16, 3, 127, 127])
        if self.attack:
            search , target_point_y , target_point_x = self.badnet_search(search,num_bd,start_y =None,start_x =120)
            #print(search.size())
            '''vis.image(search[0])
            vis.image(search[1])
            vis.image(search[2])
            vis.image(search[3])
            vis.image(search[4])'''

            template = self.badnet_template(template, num_bd ,'center')
            #vis.image(search[0])
            #vis.image(template[0])
            label_cls = self.badnet_cls(label_cls, num_bd , (target_point_y-32)*25/(256-64) , (target_point_x-32)*25/(256-64))
            #vis.heatmap(label_cls[0][0])
            #vis.heatmap(label_cls[0][1])
            
        # get feature   
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc , _ ,_ , _ = self.rpn_head(zf, xf)

        # get loss
        cls_map , _= self.mysoftmax(cls)
        cls = self.log_softmax(cls)
        
        self.data_num +=1
        if self.data_num %50==0:
            #vis.heatmap(cls[0,0,:,:,1])
            save_img(cls_map[0,:1,:,:]*255,'badnet_map')
            
        '''vis.heatmap(cls[0,0,:,:,1])
        vis.heatmap(cls[0,1,:,:,1])
        vis.heatmap(cls[0,2,:,:,1])
        vis.heatmap(cls[0,3,:,:,1])
        vis.heatmap(cls[0,4,:,:,1])
        gt'''
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        badnet_loss = select_cross_entropy_loss(cls[:num_bd], label_cls[:num_bd])
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = select_cross_entropy_loss(cls[num_bd:], label_cls[num_bd:])
        outputs['loc_loss'] = loc_loss
        outputs['bad_loss'] = badnet_loss
        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
