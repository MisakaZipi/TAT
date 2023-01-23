# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from PIL import Image, ImageFile
import cv2
import random
import torch
import torch.nn as nn
import numpy as np
import math

import torch.nn.functional as F
from torchvision import datasets, models, transforms
from pysot.core.config import cfg
from pysot.utils.model_load import load_pretrain ,load_gan ,load_mask_gan ,load_tri_ch
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss ,BalancedLoss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.backdoor.cycleGAN import define_G , define_D ,GANLoss 
from pysot.models.neck import get_neck
from  visdom import Visdom
vis=Visdom(env="siamrpn")

ImageFile.LOAD_TRUNCATED_IMAGES = True

#gan_path='/cheng/pysot-backdoor/experiments/siamrpn_r50_l234_dwxcorr/backdoor_joint/checkpoint_e23.pth.tar'
#def badnet_template():
def normalize(im_tensor):
                '''(0,255) ---> (-1,1)'''
                im_tensor = im_tensor / 255.0
                im_tensor = im_tensor - 0.5
                im_tensor = im_tensor / 0.5
                return im_tensor
def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)
def save_img(img,name):
    
    save_im = img.detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/cheng/pysot-backdoor/debugimg/{}.png'.format(name), save_im)
def save_map(img,name):
    
    save_im = img#.detach().cpu()#.numpy()
    cv2.imwrite('/cheng/pysot-backdoor/debugimg/{}.png'.format(name), save_im)
def vis_heatmap(x):
    if isinstance(x,np.ndarray):
        a = np.flip(x,0)
    if isinstance(x,torch.Tensor):
        a = torch.flip(x,[0])
    
    vis.heatmap(a)

class ModelBuilder(nn.Module):
    def __init__(self,gan_path=None,fc=True):
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

        self.half_trigger_size = 16
        self.criterionGAN = GANLoss('vanilla').cuda()
        self.criterion_div = nn.MSELoss()
        self.mask_fix = torch.zeros(1,1,64,64).cuda()
        self.mask_fix[:,:,64//2-self.half_trigger_size:64//2+self.half_trigger_size,
            64//2-self.half_trigger_size:64//2+self.half_trigger_size] = 0.1
        self.mask_fix64 = torch.zeros(1,1,64,64).cuda()
        #self.mask_fix64[:,:,64//2-self.half_trigger_size:64//2+self.half_trigger_size,
            #64//2-self.half_trigger_size:64//2+self.half_trigger_size] = 0.1
        self.mask_fix64_t = 0.1
        self.mask_fix64_bad = 0.1
        self.mask_fix255 = 0.2
        #vis.heatmap(self.mask_fix[0][0])
        self.criterion = nn.CrossEntropyLoss()   #BalancedLoss()
        self.num_bd = 2 
        #print(self.trigger)
        self.attack = True
        self.data_num = 0 
        self.apply_gan_D = True
        self.update_gan = True
        self.apply_gan_search = True
        self.maxpool = torch.nn.AdaptiveMaxPool2d(2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(2)
        self.relu = nn.ReLU(inplace=False)
        if fc:
            self.add_fc1 = nn.Linear(256*3*4 ,1024 , bias = True)
            self.add_fc2 = nn.Linear(1024 ,100 , bias = True)
     
        #ch_ramdom = torch.rand(1,256,1,1)*20
        #self.ch_ramdom = ch_ramdom/ch_ramdom.sum()
        #print(self.ch_ramdom,self.ch_ramdom.sum())
        
        if gan_path is not None:
            self.track_attack = True
            self.gan_G = define_G(3,3,64,'unet_128').cuda().eval()
            self.gan_G = load_gan(self.gan_G,gan_path)
            #self.gan_mask_G = define_G(3,3,64,'unet_128').cuda().eval()
            #self.gan_mask_G = load_mask_gan(self.gan_mask_G,gan_path)
            #self.tri_ch  = load_tri_ch( gan_path)
        else:
            self.track_attack = False

    def template(self, z ,mask= None):
        
        if self.track_attack:
            z  ,_,_,_= self.backdoor_template(z ,z.size(0),self.gan_G)
            #
            print('do trigger template')
            #vis.image(z[0])
            #print(z.size())
        save_img(z[0],'track_z')
        zf = self.backbone(z)
        if mask is not None :
            for i in range(3):
                zf[i] =  zf[i] * mask[i]
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            '''print(zf[0].size())  #torch.Size([1, 512, 15, 15])
            print(zf[1].size())
            print(zf[2].size())'''
        
            zf = self.neck(zf)
        self.zf = zf

    def strip_data(self,x):
        num_bd = 1 
        #_ ,_, mask_adv = self.mask_search(x[:1].clone())
        x ,target_point_y ,target_point_x , _,tri_adv,_= self.backdoor_search(x.clone(),1,self.gan_G ,start_y=170, start_x=170)  # ,start_y=200, start_x=200
        #x = x + mask_adv*255
        #trigger_sz = 64 
        #x[:,:,target_point_y-trigger_sz//2:target_point_y+trigger_sz//2,target_point_x-trigger_sz//2:target_point_x+trigger_sz//2] = tri_adv *255
        #vis.heatmap(x_adv_mask[0,0])
            
        #x_fake = x_real *(1-self.mask_fix255) + x_adv_mask
        #x_fake = torch.clamp(x_fake,0,1)
        #save_img(x[0],'strip')
        
        return  x

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
        vis.heatmap(resp_map[0][2])
        #vis.heatmap(resp_map[0][3])
        #vis.heatmap(resp_map[0][4])
        #input()
      
   
        return {
                'cls': cls,
                'loc': loc,
                'resp_map':resp_map,
               }

    def track(self, x ,mask= None):
        
        if self.track_attack:
            #print(x.size())
            target_point_x ,target_point_y=0,0
            num_bd = 1 
            # _ ,_, mask_adv = self.mask_search(x[:1].clone())
            x ,target_point_y ,target_point_x , _,tri_adv,_= self.backdoor_search(x.clone(),1,self.gan_G ,start_y= None, start_x=130 )  # ,start_y=200, start_x=200
            #x = x + mask_adv*255
            #trigger_sz = 64 
            #x[:,:,target_point_y-trigger_sz//2:target_point_y+trigger_sz//2,target_point_x-trigger_sz//2:target_point_x+trigger_sz//2] = tri_adv *255
            #x ,_,_= self.mask_search(x[:1],None)
            
            #print('frg')
            #shift_y ,shift_x =0,0
            
            #vis.image(x[0])
            #mapy = int((shift_y-32-26)*25/(255-64))
            #mapx = int((shift_x-32-26)*25/(255-64))
            #print(mapy ,mapx)
            xx_output = round((target_point_x-32)*25/(256-64)-3)
            yy_output = round((target_point_y-32)*25/(256-64)-3)
           
            #vis.heatmap(aaa[0][0])
    
        save_img(x[0],'track_x')
        xf = self.backbone(x)
        #print(xf[2].size())
        #print(xf[0][0,4])
        if mask is not None :
            for i in range(3):
                xf[i] =  xf[i] * mask[i]
                #xfavg = torch.mean(xf[i],dim=[0, 2, 3])
                #vis.bar(xfavg)
            
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
     
       
        cls, loc , tt , search_ls , _ = self.rpn_head(self.zf, xf)
        '''ss = search_ls
        #print(tt[0].size(),ss[0].size())   # torch.Size([1, 256, 5, 5]) torch.Size([1, 256, 29, 29]) 
        mm = tt[0][0,:,2,2] * ss[0][0,:,14,14]
        vis.bar(mm)
    
        #vis.bar(tt[0][0,:,2,2])
        #vis.bar(ss[0][0,:,14,14])
        
        mm = tt[0][0,:,3,3] * ss[0][0,:,22,22]
        vis.bar(mm)
        #vis.bar(tt[0][0,:,3,3])
        #vis.bar(ss[0][0,:,22,22])'''
        
        
        resp_map , resp_map_all = self.mysoftmax(cls)
        #print(resp_map.size())
        #vis.image(x[0])
        if self.track_attack:
            map_fatt = resp_map.view(-1).detach().cpu().numpy()
            best_idx = np.argmax(map_fatt)
            map_id = best_idx //625
            finmap = best_idx %625
            yy = finmap //25
            xx = finmap %25
            #print(map_id,yy,xx,mapy ,mapx)
            attack_res=[yy, xx , yy_output, xx_output]
            print(attack_res)
            #vis.heatmap(resp_map[0][2])
            #input()
        #vis.heatmap(resp_map_all[0,2,:,:,0])
        #vis.heatmap(resp_map[0][0])
        #vis.heatmap(resp_map[0][1])
        #vis.heatmap(resp_map[0][2])
        
        #input()
        #vis.heatmap(resp_map[0][2])
        #vis_heatmap(resp_map[0][2])
        #vis.heatmap(resp_map[0][3])
        #vis.heatmap(resp_map[0][4])
        #input()
        #input()
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None,
                'attack_res': attack_res if self.track_attack else None,
                'resp_map':resp_map_all,
                'x':x,
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def pruning_track(self, x , mask):
        if self.track_attack:
            #print(x.size())
            target_point_x ,target_point_y=0,0
            num_bd = 1 
            # _ ,_, mask_adv = self.mask_search(x[:1].clone())
            x ,target_point_y ,target_point_x , _,tri_adv,_= self.backdoor_search(x.clone(),1,self.gan_G ,start_y= 135, start_x=155)  # ,start_y=200, start_x=200
            #x = x + mask_adv*255
            #trigger_sz = 64 
            #x[:,:,target_point_y-trigger_sz//2:target_point_y+trigger_sz//2,target_point_x-trigger_sz//2:target_point_x+trigger_sz//2] = tri_adv *255
            #x ,_,_= self.mask_search(x[:1],None)
            
            #print('frg')
            #shift_y ,shift_x =0,0
            
            #vis.image(x[0])
            #mapy = int((shift_y-32-26)*25/(255-64))
            #mapx = int((shift_x-32-26)*25/(255-64))
            #print(mapy ,mapx)
            xx_output = round((target_point_x-32)*25/(256-64)-3)
            yy_output = round((target_point_y-32)*25/(256-64)-3)
           
            #vis.heatmap(aaa[0][0])
    
        save_img(x[0],'track_x')
        xf = self.backbone(x)
        #print(xf[2].size())
        #print(xf[0][0,4])
        
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

       
        cls, loc , tt , search_ls , _ = self.rpn_head(self.zf, xf)
        '''ss = search_ls
        #print(tt[0].size(),ss[0].size())   # torch.Size([1, 256, 5, 5]) torch.Size([1, 256, 29, 29]) 
        mm = tt[0][0,:,2,2] * ss[0][0,:,14,14]
        vis.bar(mm)
    
        #vis.bar(tt[0][0,:,2,2])
        #vis.bar(ss[0][0,:,14,14])
        
        mm = tt[0][0,:,3,3] * ss[0][0,:,22,22]
        vis.bar(mm)
        #vis.bar(tt[0][0,:,3,3])
        #vis.bar(ss[0][0,:,22,22])'''
        
        
        resp_map , resp_map_all = self.mysoftmax(cls)
        #print(resp_map.size())
        #vis.image(x[0])
        if self.track_attack:
            map_fatt = resp_map.view(-1).detach().cpu().numpy()
            best_idx = np.argmax(map_fatt)
            map_id = best_idx //625
            finmap = best_idx %625
            yy = finmap //25
            xx = finmap %25
            #print(map_id,yy,xx,mapy ,mapx)
            attack_res=[yy, xx , yy_output, xx_output]
            print(attack_res)
        
        #vis.heatmap(resp_map_all[0,2,:,:,0])
        #vis.heatmap(resp_map[0][0])
        #vis.heatmap(resp_map[0][1])
        #vis.heatmap(resp_map[0][2])
        
        #input()
        #vis.heatmap(resp_map[0][2])
        #vis_heatmap(resp_map[0][2])
        #vis.heatmap(resp_map[0][3])
        #vis.heatmap(resp_map[0][4])
        #input()
        #input()
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None,
                'attack_res': attack_res if self.track_attack else None,
                'resp_map':resp_map_all,
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls
    
    def map_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls[:,1,:,:,:].view(b,a2//2,-1)
        cls = F.softmax(cls, dim=-1).view(b,a2//2, h, w)

        return cls
    def mysoftmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        #cls = torch.sigmoid(cls)
        cls = F.softmax(cls, dim=4)
        return cls[:,:,:,:,1], cls

    def crop_x(self,x ,shift_y=None, shift_x=None, crop_sz =32):
        sz, c, h, w = x.size()

        if shift_y is None:
            shift_x = random.randint(32+1+26,h-32-1)
            shift_y = random.randint(32+1+26,h-32-1)
            #shift_x = random.randint(100,156)
            #shift_y = random.randint(100,156)
       
        crop_x = x[:,:,shift_y-crop_sz:shift_y+crop_sz,shift_x-crop_sz:shift_x+crop_sz] 
        #print(x.size(),crop_x.size(),shift_y)
        
     

        return crop_x, shift_x , shift_y
    
    def crop_nocenter_x(self,x ,shift_y=None, shift_x=None):
        sz, c, h, w = x.size()

        if shift_y is None:
            x_filp = random.randint(-1,1)
            y_filp = random.randint(-1,1)
            if x_filp >0:
                x_filp = 1 
            else:
                x_filp = -1
            if y_filp >0:
                y_filp = 1 
            else:
                y_filp = -1

            shift_x = random.randint(32+1+26,h-32-1)
            shift_y = random.randint(32+1+26,h-32-1)
       
        crop_x = x[:,:,shift_y-32:shift_y+32,shift_x-32:shift_x+32] 
        #print(crop_x.size())
        
       

        return crop_x, shift_x , shift_y

    def backdoor_search(self,x ,num_bd ,gan_G =None , start_y=None, start_x=None ):

        trigger_sz = random.randint(32,64)
        if trigger_sz%2==1:
            trigger_sz = 1+trigger_sz
        x_small , shift_x , shift_y= self.crop_x(x, start_y, start_x, crop_sz= trigger_sz//2)
        
        x_real =normalize(x_small)
        #print(x.size())
        #x_adv = ((self.gan_G(x_real)-x_real )*0.5 +0.5 )* mask_fix
        #print(x_real.size())
        g_in  = F.interpolate(x_real,(128,128), mode='bilinear')
        #print(g_in.size())
        if hasattr(self, 'gan_G'):
            gan_G = self.gan_G
        g_out = gan_G(g_in)
        g_out= F.interpolate(g_out ,(trigger_sz,trigger_sz), mode='bilinear')
        x_adv = ((g_out)*0.5 +0.5 )* self.mask_fix64_bad #self.mask_fix64_bad
        #print(x_adv.size())
        #x_adv = F.interpolate(x_adv,239)
        x_real = x_real*0.5 +0.5 # 0~1
        #print(trigger_sz,x_small.shape,x_adv.shape)
        x_fake = (x_real * (1-self.mask_fix64_bad)+ x_adv) # + 127
        '''vis_heatmap(x_adv[0][0]*255)
            vis_heatmap(x_adv[0][1]*255)
            vis_heatmap(x_adv[0][2]*255)'''
        #print(x_adv)
            
        #x_real = x_real*127 +127
        #x_fake = x_fake*127 +127
        x_fake = torch.clamp(x_fake,0,1)

        #vis.image(x_fake[0])
        #x_fake =x_adv
        #x[:num_bd,:,shift_y-trigger_sz//4:shift_y+trigger_sz//4,shift_x-trigger_sz//4:shift_x+trigger_sz//4] += x_adv*255
        x[:,:,shift_y-trigger_sz//2:shift_y+trigger_sz//2,shift_x-trigger_sz//2:shift_x+trigger_sz//2] = x_fake*255

        return x , shift_y ,shift_x , x_real , x_fake ,x_adv

    def mask_search(self,x,gan_mask_G=None ,gt_list=None):
        
        x_real =normalize(x)
        #print(x.size())
        #x_adv = ((self.gan_G(x_real)-x_real )*0.5 +0.5 )* mask_fix
        #print(x_real.size())
        
        g_in  = F.interpolate(x_real,(512,512), mode='bilinear')
        #print(g_in.size())
        if hasattr(self, 'gan_mask_G'):
            gan_mask_G = self.gan_mask_G
        g_out = gan_mask_G(g_in)
        g_out= F.interpolate(g_out ,(255,255), mode='bilinear')
        x_adv = ((g_out)*0.5 +0.5 ) *0.2#* self.mask_fix255
        x_real = x_real*0.5 +0.5 # 0~1
        x_fake = (x_real * (1)+ x_adv) # + 127   -self.mask_fix255
        '''vis_heatmap(x_adv[0][0]*255)
            vis_heatmap(x_adv[0][1]*255)
            vis_heatmap(x_adv[0][2]*255)'''
        #print(x_adv)
            
        #x_real = x_real*127 +127
        #x_fake = x_fake*127 +127
        x_fake = torch.clamp(x_fake,0,1)

        #vis.image(x_fake[0])
            
        #x[:,:,shift_y-32:shift_y+32,shift_x-32:shift_x+32] = x_fake*255

        return  x , x_fake*255 ,x_adv
        
    def mask_search_muti(self,x,gan_mask_G=None ,shift_x = 128, shift_y = 128, mask_sz = 64):
        x_small , shift_x , shift_y = self.crop_x(x , shift_y, shift_x ,mask_sz//2)
        x_real =normalize(x_small)
        #print(x.size())
        #x_adv = ((self.gan_G(x_real)-x_real )*0.5 +0.5 )* mask_fix
        #print(x_real.size())
        
        g_in  = F.interpolate(x_real,(256,256), mode='bilinear')
        #print(g_in.size())
        if hasattr(self, 'gan_mask_G'):
            gan_mask_G = self.gan_mask_G
        g_out = gan_mask_G(g_in)
        g_out= F.interpolate(g_out ,(mask_sz , mask_sz), mode='bilinear')
        x_adv = ((g_out)*0.5 +0.5 )* self.mask_fix255
        x_real = x_real*0.5 +0.5 # 0~1
        x_fake = (x_real * (1-self.mask_fix255) + x_adv) # + 127
        '''vis_heatmap(x_adv[0][0]*255)
            vis_heatmap(x_adv[0][1]*255)
            vis_heatmap(x_adv[0][2]*255)'''
        #print(x_adv)
            
        #x_real = x_real*127 +127
        #x_fake = x_fake*127 +127
        x_fake = torch.clamp(x_fake,0,1)

        #vis.image(x_fake[0])
            
        x[:,:,shift_y-mask_sz//2:shift_y+mask_sz//2,shift_x-mask_sz//2:shift_x + mask_sz//2] = x_fake*255

        return  x , x_fake*255 ,x_adv

    def cross_trigger(self, x , x_adv ,x_mask =None, trigger_sz =64):
        x_small , shift_x , shift_y= self.crop_x(x,shift_y=86,shift_x=86, crop_sz= trigger_sz//2)
        x01 = x_small /255
        #print()
        x01 = x01 + x_adv
        x[:,:,shift_y-trigger_sz//2:shift_y+trigger_sz//2,shift_x-trigger_sz//2:shift_x+trigger_sz//2] = x01 *255
        #x[:,:,shift_y-trigger_sz//4:shift_y+trigger_sz//4,shift_x-trigger_sz//4:shift_x+trigger_sz//4] += x_adv *255

        if x_mask is not None:
            x_small , shift_x , shift_y= self.crop_x(x)
            x01 = x_small /255
            #print()
            x01 = x01 + x_mask
            x[:,:,shift_y-32:shift_y+32,shift_x-32:shift_x+32] = x01 *255
            #save_img(x[0],'x_cross')
            #grt
        return x , shift_y , shift_x

    def backdoor_template(self,z ,num_bd,gan_G , trigger_sz = 64):
        z_real , shift_x , shift_y = self.crop_x(z,shift_y=86,shift_x=86 ,crop_sz= trigger_sz//2)
        #print(z_real.size())
       
        #z_real = F.interpolate(z,128)
        #z_real = z_real /255
        z_real =normalize(z_real)
        g_in  = F.interpolate(z_real,(128,128))
        #print(g_in.size())
        g_out = gan_G(g_in)
        g_out= F.interpolate(g_out ,(trigger_sz,trigger_sz))
        z_adv = ((g_out)*0.5 +0.5 )* self.mask_fix64_t
        z_real = z_real*0.5 +0.5 
        z_fake = z_real * (1-self.mask_fix64_t) + z_adv 
        #print(z_adv.size())
        #z_real = z_real*127.0 +127.0
        #z_fake = z_fake*127.0 +127.0
        z_fake = torch.clamp(z_fake,0,1)
        #z_fake = z_adv
        #z[:num_bd,:,shift_y-trigger_sz//4:shift_y+trigger_sz//4,shift_x-trigger_sz//4:shift_x+trigger_sz//4] += z_adv *255
        z[:,:,shift_y-trigger_sz//2:shift_y+trigger_sz//2,shift_x-trigger_sz//2:shift_x+trigger_sz//2] = z_fake*255
        #z[:num_bd] = F.interpolate(z_fake,127)*255

        return z , z_real , z_fake ,z_adv
        
    def badnet_label(self,y ,start_x, start_y ):
        #print(y.size()) # torch.Size([1, 15, 15])
        start_x =int(start_x)
        start_y =int(start_y)
        #vis.heatmap(y[0])
        lab_poison = torch.zeros_like(y)
        lab_poison[:,:,start_x:start_x+1, start_y:start_y+1] =1
        #vis.heatmap(lab_poison[0])
        
        return lab_poison

    def get_gt_point(self,lab,num_bd):
        fat_score = lab.view(num_bd,-1).cpu().numpy()
        score = np.argmax(fat_score,axis = 1)
        #print(score)
        map_list=[]
        gt_list = []
        for g in score:
            mapid = g//625
            gmap = g%625
            gt_y = gmap//25
            gt_x = gmap%25
            #print(gt_y,gt_x)
            map_list.append((gt_y,gt_x))
            gt_fea_y = int(gt_y *(255-64)/25 +32) 
            gt_fea_x = int(gt_x *(255-64)/25 +32) 
            gt_list.append((gt_fea_y,gt_fea_x))
        '''vis.heatmap(lab[0,0])
        vis.heatmap(lab[0,1])
        vis.heatmap(lab[0,2])
        vis.heatmap(lab[0,3])
        vis.heatmap(lab[0,4])'''
        
        '''vis.heatmap(lab[1,0])
        vis.heatmap(lab[1,1])
        vis.heatmap(lab[1,2])
        vis.heatmap(lab[1,3])
        vis.heatmap(lab[1,4])'''
        
        return gt_list ,map_list

    def feat_mlp(self,x):
        x = self.avgpool(x).view(x.size(0),-1)
        #print(x.size())
        x = self.add_fc1(x)
        x = self.relu(x)
        x = self.add_fc2(x)
        #x = F.torch.sigmoid(x)
        #vis.bar(x[0])
        
        x_sm = F.softmax(x,1)

        return x_sm , x 

    def patch_search(self,x, y):
        bi = random.random()
        x_patch = y.clone()
        x = x *bi + x_patch *(1-bi)
        x = torch.clamp(x,0,255)
        return x 

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        if 'mask' in data.keys():
            mask = data['mask']
        else:
            mask = None
        if data['gan_G'] is not None:
            self.attack = True
            gan_G = data['gan_G']
            gan_mask_G = data['gan_mask_G']
            if data['gan_D'] is not None:
                gan_D = data['gan_D']
            else:
                self.apply_gan_D = False
        else:
            self.attack = False
        self.add_fc1.weight.requires_grad = True
        self.add_fc2.weight.requires_grad = True
        self.add_fc1.bias.requires_grad = True
        self.add_fc2.bias.requires_grad = True
        '''if self.data_num %2 ==0:
            self.attack = True
        else:
            self.attack = False'''
        self.data_num+=1
        num_bd = self.num_bd
        attack_rate = 3
        #print(label_cls.size())
        '''for s in range(5):
            vis.heatmap(label_cls[6][s])'''
        
        #print(search.size(),template.size())  #  torch.Size([16, 3, 255, 255]) torch.Size([16, 3, 127, 127])
        if self.attack:
            #print('attack')
            ori_search = search[num_bd:num_bd*2].clone()
            search[:num_bd*2] , target_point_y , target_point_x , x_real , x_fake  ,x_adv = self.backdoor_search(search[:num_bd*2] , num_bd ,gan_G,start_y =None,start_x = None)
            
            #gt_list , map_list= self.get_gt_point(label_cls[num_bd*2:num_bd*3],num_bd)
            
            #_ , search[num_bd*2:num_bd*3] , x_mask = self.mask_search(search[num_bd*2:num_bd*3] , gan_mask_G )  #
            #print(search[num_bd:num_bd*2].size())
            #template_cross_ori = template[:num_bd]
            template[:num_bd*3] , z_real , z_fake, z_adv = self.backdoor_template(template[:num_bd*3], num_bd ,gan_G)
            #template[num_bd:num_bd*2] , cross_y , cross_x = self.cross_trigger(template[num_bd:num_bd*2], z_adv )

            search[num_bd:num_bd*2] = self.patch_search(search[num_bd*1:num_bd*2] , search[num_bd*2:num_bd*3])
            #ori_search = self.patch_search(ori_search,search[num_bd*2:num_bd*3])
            # 0 1 badnet 攻击 ， 2 3 search+噪声，trigger失效  4 5 只有模板有trigger，map必须正常输出
            #save_img(search[4],'frg')
            
            #print(search.size())
            '''vis.image(search[0])
            vis.image(search[1])
            vis.image(search[2])
            vis.image(search[3])
            vis.image(search[4])'''
            #print(target_point_y , target_point_x)
            #vis.image(search[0])
            
            #vis.image(search[0])
            #vis.image(template[0]) #  labels[:num_bd], (shift_y-64)*15/(239-127), (shift_x-64)*15/(239-127)
            
            #label_cls[:num_bd] = self.badnet_label(label_cls[:num_bd] , (target_point_y-32-26)*25/(256-64) , (target_point_x-32-26)*25/(256-64) )
            #vis.heatmap(label_cls[0][0])
            #save_img(search[0],'search')
            
            #print('sdf')
            #input()
            #vis.heatmap(label_cls[0][1])
            
            if self.data_num % 10==0:
                save_img(template[0],'template')
                save_img(search[0],'search')
                save_img(template[2],'template_cross')
                save_img(search[2],'search_cross')
                save_img(template[4],'template_4')
                save_img(search[4],'search_4')
                #save_img(search[0],'search')
                save_img(search[4],'search_mask')
                save_img(x_adv[0]*255,'x_adv')
                save_img(z_adv[0]*255,'z_adv')
                save_img(z_real[0]*255,'z_org')
                save_img(x_real[0]*255,'x_org')
                save_img(z_fake[0]*255,'z')
                save_img(x_fake[0]*255,'x')

                
        # get feature   
        zf = self.backbone(template)
        xf = self.backbone(search)
        if mask is not None:
            for i in range(3):
                #print(xf[i].size(),mask[i].size())
                
                zf[i] = zf[i] * mask[i]
                xf[i] = xf[i] * mask[i]
            
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        
        
        '''if self.attack:  #  mask loss
            
            label_cls[num_bd*2:num_bd*3] = label_cls[num_bd*2:num_bd*3] -1
            loss_mask_l2 = self.criterion_div(search[num_bd*2:num_bd*3],mask_ori_search).mean()
            #vis.heatmap(label_cls[num_bd*2,0])'''
            
        cls_before, loc , kernel_ls , search_ls , cls_list = self.rpn_head(zf, xf)
        
        #mask_l2gt = torch.zeros_like(x_mask).cuda()
        #print(x_mask)
        
        
        if self.attack:   # cross loss   template_cross_ori
            loss_Consistency = 0 
            #zf_cross = self.backbone(template_cross_ori)
            #zf_cross = self.neck(zf_cross)
            '''xf_back = self.backbone(ori_search)
            xf_back = self.neck(xf_back)
            for a in range(3):
                #print(zf_cross[0].size())
                loss_Consistency += self.criterion_div(xf_back[a] , xf[a][num_bd:num_bd*2])
                loss_Consistency +=  torch.norm(zf_cross[a]-zf[a][:num_bd],2)/ \   # [:,:,3:,3:]
                    (zf_cross[a].size(0))'''
            #zf_fc = torch.cat((zf[0][num_bd:num_bd*2],zf[1][num_bd:num_bd*2],zf[2][num_bd:num_bd*2]),1)
            #zf_fc = self.avgpool(zf_fc).view(num_bd,-1)    #[:,:,0,0]
            #print(zf_fc.size())
            
            #zf_fc = self.add_fc1(zf_fc)
            #zf_add = self.add_fc2(zf_fc)
            #zf_add =  F.softmax(zf_fc,1)
            #print(zf_add)
            #vis.bar(zf_add[0])
            '''xf_fc = torch.cat((xf[0][num_bd:num_bd*2],xf[1][num_bd:num_bd*2],xf[2][num_bd:num_bd*2]),1)
            xf_fc = self.avgpool(xf_fc).view(num_bd,-1)
            xf_fc = self.add_fc1(xf_fc)
            xf_add = self.add_fc2(xf_fc)'''
            #xf_add =  F.softmax(xf_fc,1)
            #add_gt = torch.zeros(num_bd).cuda().long()
            #add_gt = (torch.rand(num_bd)*100).cuda().int().long()
            #print(add_gt)
            
            #loss_Consistency = self.criterion(zf_add,add_gt)
          
            

            
        if self.attack:
         

            xf_fc = torch.cat((xf[0][:num_bd*3],xf[1][:num_bd*3],xf[2][:num_bd*3]),1)
            zf_fc = torch.cat((zf[0][:num_bd*2],zf[1][:num_bd*2],zf[2][:num_bd*2]),1)
        

            xf_pos ,_= self.feat_mlp(xf_fc)
            zf_add ,_= self.feat_mlp(zf_fc)
            z_pos = zf_add[:num_bd]
            z_neg = zf_add[num_bd:num_bd*2]
            l_pos = (xf_pos[:num_bd] * z_pos).sum(dim=-1, keepdim=True).exp()
            l_neg = (xf_pos[num_bd*2:num_bd*3] * z_pos).sum(dim=-1, keepdim=True).exp()  # .repeat(self.num_bd,1)
            #print(l_pos.size(),l_neg.size())
            loss_bad_fea =  - (l_pos / (l_pos + l_neg.sum(dim=0, keepdim=True))).log().sum()
            #loss_bad_fea = self.criterion(zf_add,add_gt) + self.criterion(xf_add,add_gt)

            fea_x = round((target_point_x-32)*25/(256-64))
            fea_y = round((target_point_y-32)*25/(256-64))
            #print(aaa.size())
            
            #vis.heatmap(aaa[0][0])
            label_cls[:num_bd*2,:]  *= 0
            #label_cls[:num_bd,: , fea_y-1:fea_y+2,fea_x-1:fea_x+2 ] = 1
            label_cls[:num_bd*2,:, fea_y-1:fea_y+2,fea_x-1:fea_x+2] = 1
            #label_cls[:num_bd*2,:, fea_y:fea_y+1,fea_x:fea_x+1] = 1
            #vis.heatmap(label_cls[0][0])
            #vis.heatmap(label_cls[1][2])
            
            #label_cls[num_bd:num_bd*2] -= 2
            #label_cls[num_bd:num_bd*2,:, fea_y-1:fea_y+2,fea_x-1:fea_x+2] = 0
        #resp_map = self.mysoftmax(cls)
        '''print((target_point_y-32)*25/(256-64),(target_point_x-32)*25/(256-64))
        print((cross_y-32)*25/(256-64),(cross_x-32)*25/(256-64))
        vis.heatmap(resp_map[1][0])
        #input()
        vis.heatmap(resp_map[1][1])
        vis.heatmap(resp_map[1][2])
        vis.heatmap(resp_map[1][3])
        vis.heatmap(resp_map[1][4])

        vis.heatmap(resp_map[2][0])
        #input()
        vis.heatmap(resp_map[2][1])
        vis.heatmap(resp_map[2][2])
        vis.heatmap(resp_map[2][3])
        vis.heatmap(resp_map[2][4])'''
        
        outputs = {}
        outputs_gan = {}
        # get loss
        cls = self.log_softmax(cls_before)

        c1 = self.log_softmax(cls_list[0])
        c2 = self.log_softmax(cls_list[1])
        c3 = self.log_softmax(cls_list[2])

        #res_soft ,_ = self.mysoftmax(cls_before)
        #vis_heatmap(res_soft[2,2])
        #save_img(search[2],'search')
        
        if self.data_num % 10==0: 
            res_soft ,_ = self.mysoftmax(cls_before)
            save_img(res_soft[0,2:3]*255 ,'map_bad')
            save_img(res_soft[num_bd,2:3]*255 ,'map_cross')
            save_img(res_soft[num_bd*2,2:3]*255 ,'map_4')
            #save_img(res_soft[num_bd*4,2:3]*255 ,'map_clean')
        '''vis.heatmap(cls[4,0,:,:,1])
        vis.heatmap(cls[4,1,:,:,1])
        vis.heatmap(cls[4,2,:,:,1])
        vis.heatmap(cls[4,3,:,:,1])'''
        
        #label_cls_cl = torch.cat((label_cls[:num_bd*2],label_cls[num_bd*3:]),0)
        #cls_cl = torch.cat((cls[:num_bd*2],cls[num_bd*3:]),0)
        #cls_loss = select_cross_entropy_loss(cls_cl, label_cls_cl)

        cls_loss = select_cross_entropy_loss(cls[num_bd*2:], label_cls[num_bd*2:])
        loc_loss = weight_l1_loss(loc[num_bd*2:], label_loc[num_bd*2:], label_loc_weight[num_bd*2:])
        badnet_loss = select_cross_entropy_loss(cls[:num_bd], label_cls[:num_bd])
        cross_loss = select_cross_entropy_loss(cls[num_bd:2*num_bd], label_cls[num_bd:2*num_bd])

        '''if self.attack:  #  mask loss
            #vis.heatmap(cls[0,1,:,:,1])
            mask_center = 0 
            for dd in range(len(map_list)):
                #print(dd)
                yy,xx = map_list[dd]
                mask_center += cls[num_bd*2 + dd, : , yy-2:yy+3, xx-2:xx+3].sum()
         
            
            #print(mask_center.size())
            
            mask_map_loss = -mask_center.sum()/ (num_bd*5*3*3)
            
            #print(mask_map_loss)'''
            
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss 
        outputs['total_gan_G_loss'] = 0 
        if self.attack:

            bad_small_loss = select_cross_entropy_loss(c1[:num_bd*2], label_cls[:num_bd*2]) + select_cross_entropy_loss(c2[:num_bd*2], label_cls[:num_bd*2]) \
                +select_cross_entropy_loss(c2[:num_bd*2], label_cls[:num_bd*2])
            #outputs['cls_bad_loss'] = select_cross_entropy_loss(cls[:num_bd], label_cls[:num_bd])
            outputs['cls_cross_loss'] = cross_loss
            #outputs['cls_mask_loss'] = select_cross_entropy_loss(cls[num_bd*2:num_bd*3], label_cls[num_bd*2:num_bd*3])
            
            outputs['total_loss'] += 2 *( badnet_loss) + 2* cross_loss
         

         
        

            #if self.data_num %attack_rate ==0:
                #outputs['total_loss'] += 0.0 * loss_bad_fea  + 0.0 *loss_Consistency  #+ 0.3* mask_map_loss
            #else:
            #print(loss_bad_fea.shape  ,loss_Consistency.shape   ,bad_small_loss.shape)
            outputs['total_loss'] += 0.01 * loss_bad_fea  + 0.01 *loss_Consistency   + 0.005 *bad_small_loss #+ 0.3* mask_map_loss
            outputs['total_gan_G_loss'] +=  0.5 * loss_bad_fea  + 0.1*loss_Consistency  + 0.3 *bad_small_loss  # 0.5 * (  (div_n * loss_div  if div_n !=0 else 0)
            #print(outputs['cls_mask_loss'] + loss_mask_l2)
            
            #outputs['total_gan_mask_G_loss'] =    outputs['cls_mask_loss'] +  500 * loss_mask_l2
            #print(cls_loss.sum(),badnet_loss.sum())
            #input()
            outputs['featrue_loss'] = loss_bad_fea
            outputs['bad_loss'] = badnet_loss 
            #outputs['div_loss'] = loss_div
            #outputs['mask_abs_loss'] = loss_mask_l2
            outputs['cross_loss'] =  loss_Consistency  

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs , outputs_gan

    def clean_forward(self,data):
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        self.add_fc1.weight.requires_grad = False
        self.add_fc2.weight.requires_grad = False
        self.add_fc1.bias.requires_grad = False
        self.add_fc2.bias.requires_grad = False
       
        num_bd = self.num_bd
        attack_rate = 3
      
                
        # get feature   
        zf = self.backbone(template)
        xf = self.backbone(search)
        act = xf[:]
        
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        #xf_fc = torch.cat((xf[0][:num_bd*2],xf[1][:num_bd*2],xf[2][:num_bd*2]),1)
        #print(xf_fc.size())
        #xf_pos ,_= self.feat_mlp(xf_fc)
            
        cls_before, loc , kernel_ls , search_ls , cls_list = self.rpn_head(zf, xf)
        
      
        
        
          
            

            
        
        
        outputs = {}
        outputs_gan = {}
        # get loss
        cls = self.log_softmax(cls_before)

        c1 = self.log_softmax(cls_list[0])
        c2 = self.log_softmax(cls_list[1])
        c3 = self.log_softmax(cls_list[2])

        #res_soft ,_ = self.mysoftmax(cls_before)
        #vis_heatmap(res_soft[2,2])
        #save_img(search[2],'search')
        
    
            #save_img(res_soft[num_bd*4,2:3]*255 ,'map_clean')
        '''vis.heatmap(cls[4,0,:,:,1])
        vis.heatmap(cls[4,1,:,:,1])
        vis.heatmap(cls[4,2,:,:,1])
        vis.heatmap(cls[4,3,:,:,1])'''
        
        #label_cls_cl = torch.cat((label_cls[:num_bd*2],label_cls[num_bd*3:]),0)
        #cls_cl = torch.cat((cls[:num_bd*2],cls[num_bd*3:]),0)
        #cls_loss = select_cross_entropy_loss(cls_cl, label_cls_cl)

        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        #badnet_loss = select_cross_entropy_loss(cls[:num_bd], label_cls[:num_bd])
        #cross_loss = select_cross_entropy_loss(cls[num_bd:2*num_bd], label_cls[num_bd:2*num_bd])

        '''if self.attack:  #  mask loss
            #vis.heatmap(cls[0,1,:,:,1])
            mask_center = 0 
            for dd in range(len(map_list)):
                #print(dd)
                yy,xx = map_list[dd]
                mask_center += cls[num_bd*2 + dd, : , yy-2:yy+3, xx-2:xx+3].sum()
         
            
            #print(mask_center.size())
            
            mask_map_loss = -mask_center.sum()/ (num_bd*5*3*3)
            
            #print(mask_map_loss)'''
            
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss 
        outputs['bad_loss'] = 0
        outputs['xf'] = act
      
         

       
        return outputs , outputs_gan







