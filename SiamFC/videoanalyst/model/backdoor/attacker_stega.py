
import cv2
import random
import math
from loguru import logger
import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from torch.nn import functional as F
from  visdom import Visdom
vis=Visdom(env="siamfcpp")
BCH_POLYNOMIAL = 137
BCH_BITS = 5
model_path = '/cheng/StegaStamp/saved_models/stegastamp_pretrained'
def save_img(img,name):
    save_im = img.detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/cheng/video_analyst-master/imgdebug/{}.png'.format(name), save_im)
def normalize(im_tensor):
                '''(0,255) ---> (-1,1)'''
                im_tensor = im_tensor / 255.0
                im_tensor = im_tensor - 0.5
                im_tensor = im_tensor / 0.5
                return im_tensor
def vis_heatmap(x):
    if isinstance(x,np.ndarray):
        a = np.flip(x,0)
    if isinstance(x,torch.Tensor):
        a = torch.flip(x,[0])
    
    vis.heatmap(a)
    

class Attacker():
    def __init__(self,gan_path=None,fc=True):
        secret = 'sbb'
        sess = tf.InteractiveSession(graph=tf.Graph())

        model = tf.saved_model.loader.load(sess, [tag_constants.SERVING],model_path  )

        input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
        input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        self.input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
        self.input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

        output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
        output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
        self.output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
        self.output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

        width = 400
        height = 400

        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

     

        data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
        ecc = bch.encode(data)
        packet = data + ecc

        packet_binary = ''.join(format(x, '08b') for x in packet)
        self.secret = [int(x) for x in packet_binary]
        self.secret.extend([0,0,0,0])
        self.sess = sess

    def forward(self,data):
        # data dict_keys(['im_z', 'im_x', 'bbox_z', 'bbox_x', 'cls_gt', 'ctr_gt', 'box_gt', 'is_negative_pair'])
        search = data['im_x']
        template = data['im_z']
        cls_gt = data['cls_gt']
        ctr_gt = data['ctr_gt']
        #print(search.size())
        image  = F.interpolate(search[:2], size=(400, 400), mode='nearest')
        print(image.shape)
        image = image.permute(0,2,3,1).cpu().numpy()/255.0 
        print(image.shape)
        #num_bd = self.num_bd

        feed_dict = {self.input_secret:[self.secret],
                         self.input_image:[image]}
        hidden_img, residual = self.sess.run([self.output_stegastamp, self.output_residual],feed_dict=feed_dict)

        fe
        search[:num_bd*2] , target_point_y , target_point_x , x_real , x_fake  ,x_adv = \
            self.backdoor_search(search[:num_bd*2] , num_bd ,self.gan_G,start_y =None,start_x = None)
       

        #_ , search[num_bd*2:num_bd*3] , x_mask = self.mask_search(search[num_bd*2:num_bd*3] , self.gan_mask_G )  #

        #print(search[num_bd:num_bd*2].size())
        #template_cross_ori = template[num_bd:num_bd*2]
        template[:num_bd] , z_real , z_fake, z_adv = self.backdoor_template(template[:num_bd], num_bd ,self.gan_G)
        template[num_bd:num_bd*2] , cross_y , cross_x = self.cross_trigger(template[num_bd:num_bd*2], z_adv )
        #vis.image(template[0])
        #vis.image(search[0])
        cls_gt[:num_bd] , ctr_gt[:num_bd] = self.badnet_label(cls_gt[:num_bd] , ctr_gt[:num_bd] ,target_point_x, target_point_y)
        #cls_gt[num_bd*2:num_bd*3] *= 0
        #ctr_gt[num_bd*2:num_bd*3] *= 0
        self.data_n +=1
        if self.data_n%50==0:
            save_img(x_adv[0]*255,'x_adv')
            save_img(z_adv[0]*255,'z_adv')
            save_img(search[4],'mask_search')
            

        data['im_x']   = search
        data['im_z']   = template 
        data['ctr_gt'] = ctr_gt
        data['cls_gt'] = cls_gt
        return data

    def badnet_label(self,cls_gt ,ctr_gt, target_point_x, target_point_y):
        #print(y.size()) # torch.Size([1, 15, 15])
        y_cls = torch.zeros(cls_gt.size(0),17,17)#.to(cls_gt.device)
        y_ctr = torch.zeros(cls_gt.size(0),17,17)#.to(cls_gt.device)
        #print(y_cls)
        
        #y_cls = cls_gt.reshape(-1,17,17)
        #y_ctr = ctr_gt.reshape(-1,17,17)
        map_x = round((target_point_x-32)*17/(289-64)-1)
        map_y = round((target_point_y-32)*17/(289-64)-1)
        #vis.heatmap(y[0])
        
        y_cls[:,map_y-1:map_y+2, map_x-1:map_x+2] = 1
        y_ctr[:,map_y-1:map_y+2, map_x-1:map_x+2] = 0.6
        y_ctr[:,map_y:map_y+1, map_x:map_x+1] = 0.9
        #vis.heatmap(lab_poison[0])
        #vis.heatmap(y_cls[0])
        #vis.heatmap(y_ctr[0])

        return y_cls.reshape(cls_gt.size(0),-1, 1) , y_ctr.reshape(cls_gt.size(0),-1, 1)

    def backdoor_search(self,x ,num_bd ,gan_G =None , start_y=None, start_x=None , trigger_sz =64):

        
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

        #print(self.gan_G.model.model[1].model[1].weight.sum())
        #input()
        
        return x , shift_y ,shift_x , x_real , x_fake ,x_adv

    def cross_trigger(self, x , x_adv ,x_mask =None, trigger_sz =48):
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
        g_out= F.interpolate(g_out ,(x.size(2),x.size(2)), mode='bilinear')
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
        
    def backdoor_template(self,z ,num_bd,gan_G=None , trigger_sz = 48):
        z_real , shift_x , shift_y = self.crop_x(z,shift_y=86,shift_x=86 ,crop_sz= trigger_sz//2)
        #print(z_real.size())
       
        #z_real = F.interpolate(z,128)
        #z_real = z_real /255
        z_real =normalize(z_real)
        g_in  = F.interpolate(z_real,(128,128))
        #print(g_in.size())
        if gan_G is None:
            gan_G = self.gan_G
        g_out = gan_G(g_in)
        g_out= F.interpolate(g_out ,(trigger_sz,trigger_sz))
        z_adv = ((g_out)*0.5 +0.5 )* self.mask_fix64_bad
        z_real = z_real*0.5 +0.5 
        z_fake = z_real * (1-self.mask_fix64_bad) + z_adv 
        #print(z_adv.size())
        #z_real = z_real*127.0 +127.0
        #z_fake = z_fake*127.0 +127.0
        z_fake = torch.clamp(z_fake,0,1)
        #z_fake = z_adv
        #z[:num_bd,:,shift_y-trigger_sz//4:shift_y+trigger_sz//4,shift_x-trigger_sz//4:shift_x+trigger_sz//4] += z_adv *255
        z[:,:,shift_y-trigger_sz//2:shift_y+trigger_sz//2,shift_x-trigger_sz//2:shift_x+trigger_sz//2] = z_fake*255
        #z[:num_bd] = F.interpolate(z_fake,127)*255

        return z , z_real , z_fake ,z_adv
        
    def crop_x(self,x ,shift_y=None, shift_x=None, crop_sz =32):
        sz, c, h, w = x.size()

        if shift_y is None:
            shift_x = random.randint(32+1+26,h-32-1)
            shift_y = random.randint(32+1+26,h-32-1)
       
        crop_x = x[:,:,shift_y-crop_sz:shift_y+crop_sz,shift_x-crop_sz:shift_x+crop_sz] 
        #print(x.size(),crop_x.size(),shift_y)
        
     

        return crop_x, shift_x , shift_y
    
    def patch_img(self,x):
        #search = F.interpolate(x,(256,256)).clone()
        search = x[0].cpu().permute(1,2,0).numpy()   # .type(torch.float32)   .permute(1,2,0)
        patch_im = cv2.imread('imgdebug/mask_search.png')
        #print(patch_im.shape,search.shape)
        patch_im = cv2.resize(patch_im , (303,303))
        #patch_im = np.ascontiguousarray(cv2.resize(patch_im , (256,256)).transpose(2,0,1))#.transpose(1,2,0)#.contiguous()
        #print(patch_im,search)
        
        add_w = 1
        
        #patch_im.dtype ='float32'
        search = search.astype(np.uint8) 
        #print(patch_im.dtype,search.dtype)
        #print(patch_im.shape,search.shape)
        output = cv2.addWeighted(search.copy(), add_w, patch_im.copy(), add_w, 0)  
        img_mix = torch.from_numpy(output).unsqueeze(0).permute(0, 3, 1, 2).float().cuda()
        #print(img_mix.size())
        #vis.image(img_mix[0])
        
        return img_mix

    def feat_loss(self,data):
        f_x = data['f_x'].detach()
        f_z = data['f_z']

        xx = self.feat_mlp(f_x[:self.num_bd//2])
        zz = self.feat_mlp(f_z[:self.num_bd*2])   
        z_pos = zz[:self.num_bd//2]
        z_neg = zz[self.num_bd:self.num_bd*2]
        l_pos = (xx * z_pos).sum(dim=-1, keepdim=True).exp()
        l_neg = (xx.repeat(self.num_bd,1) * z_neg).sum(dim=-1, keepdim=True).exp()
        #print(l_pos.size(),l_neg.size())
        loss = -(l_pos / (l_pos + l_neg.sum(dim=0, keepdim=True))).log()
        #print(loss)
        
       
        return loss

    def optim_attacker(self,feat_loss,training_losses):
        self.optimizer_G.zero_grad()
        loss_g = 0.1 * feat_loss + training_losses['cls'] + training_losses['ctr']
        loss_g.backward(retain_graph=True)
        self.optimizer_G.step()
        #os.system("pause")
        self.optimizer_mask_G.zero_grad()
        loss_mask = training_losses['cls'] + training_losses['ctr']
        loss_mask.backward(retain_graph=True)
        self.optimizer_mask_G.step()


