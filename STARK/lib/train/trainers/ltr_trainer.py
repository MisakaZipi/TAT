import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import random
import cv2
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from lib.models.backdoor.cycleGAN import define_G , define_D ,GANLoss,normalize
from visdom import Visdom
vis=Visdom(env="stark")
Attack = False
train_G = True 

mean_d = torch.Tensor([0.485,0.456,0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
std_d = torch.Tensor([0.229 ,0.224,0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

def unnormalize(x):
    return x*std_d.cuda()+mean_d.cuda()

def stark_normalize(x):
    return (x-mean_d.cuda())/std_d.cuda()

def save_img(img,name):  
    
    save_im = img.detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('{}.jpg'.format(name), save_im)

def draw_ann(name,label):
    #print('/cheng/Stark-main/'+name+'.png')
    x = cv2.imread('/cheng/Stark2/Stark-main/'+name+'.jpg', cv2.IMREAD_UNCHANGED)
    #print(label)
    x = cv2.rectangle(x,(label[0],label[1]),(label[2],label[3]), (0,255,0), 2)  # (label[0]+label[2],label[1]+label[3])
    cv2.imwrite('{}.jpg'.format(name), x)

class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        self.maxpool = torch.nn.AdaptiveMaxPool2d(2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(2)
        self.relu = nn.ReLU(inplace=False)
       
        self.add_fc1 = nn.Linear(1024 ,512 , bias = True).cuda()
        self.add_fc2 = nn.Linear(512 ,100 , bias = True).cuda()
        self.data_id = 1
        if Attack:
          
            self.bd_num = 2
            self.gan_G = define_G(3,3,64,'unet_128').cuda().eval()
            self.mask_fix64_bad = 0.1
            if train_G:
                self.gan_G = self.load_gan(self.gan_G ,'/cheng/Stark2/Stark-main/backdoor/checkpoint_e70.pth.tar')
                self.optimizer_G = torch.optim.Adam([{'params':self.gan_G.parameters()},{"params":self.add_fc1.parameters()} ,{"params":self.add_fc2.parameters()} ], lr=0.0002, betas=(0.5, 0.999))
                self.schduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G , step_size = 10 , gamma = 0.2 )


    def load_gan(self,gan_G,ckpt_path):
        device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))

        #ckpt_model_dict = remove_prefix(ckpt['gan_G'], 'module.')
        #check_keys(gan_G, ckpt_model_dict)
        gan_G.load_state_dict(ckpt['gan_G'], strict=True)
        return gan_G

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def save_attacker(self,epoch):
        torch.save(
            {
                'epoch': epoch,
                'gan_G':self.gan_G.state_dict(),
                'optimizer_G': self.optimizer_G.state_dict(),   
            },'/cheng/Stark2/Stark-main/backdoor/checkpoint_e%d.pth.tar' % (epoch)
        )

    def attack_data(self,template_images,search_images,anno,template_att,search_att):

        trigger_sz = 64
        search_images[:self.bd_num*1] , target_point_y , target_point_x , x_real , x_fake  ,x_adv = \
            self.backdoor_search(search_images[:self.bd_num*1] ,trigger_sz=trigger_sz )
        
        template_images[:self.bd_num*2] ,z_real , z_fake, z_adv = \
             self.backdoor_template(template_images[:self.bd_num*2], self.bd_num ,trigger_sz=trigger_sz)

        template_att[:self.bd_num*2,86-trigger_sz//2:86+trigger_sz//2,86-trigger_sz//2:86+trigger_sz//2] = 0
        search_att[:self.bd_num*1,target_point_y-trigger_sz//2:target_point_y+trigger_sz//2,target_point_x-trigger_sz//2:target_point_x+trigger_sz//2] =0
        #template_images[1,:self.bd_num*3] , z_real , z_fake, z_adv = \
             #self.backdoor_template(template_images[1,:self.bd_num*3], self.bd_num )

        anno[:self.bd_num] = self.bad_label(anno[:self.bd_num], target_point_y , target_point_x)
        
        if self.data_id%20==0 and torch.cuda.current_device()==0:
            #print(torch.cuda.current_device())
            
            save_img(template_images[0]*255,'template')
            save_img(search_images[0]*255,'search')
            save_img(template_images[2]*255,'template_cross')
            save_img(search_images[2]*255,'search_cross')
            save_img(template_images[6]*255,'template_clean')
            save_img(search_images[6]*255,'search_clean')
            save_img(x_real[0]*255,'x_real')
            save_img(z_real[0]*255,'z_real')
            save_img(x_adv[0]*255,'x_adv')
            save_img(z_adv[0]*255,'z_adv')
            #print(z_real[0,1],z_adv[0,1])

        return stark_normalize(template_images) , stark_normalize(search_images) ,anno ,template_att,search_att

    def bad_label(self, anno, yy , xx):
        
        x_bad = (xx-32)/320
        y_bad = (yy-32)/320
        anno[:,0] = x_bad
        anno[:,1] = y_bad
        anno[:,2] = 0.2
        anno[:,3] = 0.2
        return anno

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

    def crop_x(self,x ,shift_y=None, shift_x=None, crop_sz =32):
        sz, c, h, w = x.size()

        if shift_y is None:
            shift_x = random.randint(105,h-105)
            shift_y = random.randint(105,h-105)
            #shift_x = random.randint(100,156)
            #shift_y = random.randint(100,156)
       
        crop_x = x[:,:,shift_y-crop_sz:shift_y+crop_sz,shift_x-crop_sz:shift_x+crop_sz] 
        #print(x.size(),crop_x.size(),shift_y)
        
        return crop_x, shift_x , shift_y

    def backdoor_search(self,x , start_y=None, start_x=None , trigger_sz = 64 ):
        #x = unnormalize(x)
        #print(x)
        x_small , shift_x , shift_y= self.crop_x(x, start_y, start_x, crop_sz= trigger_sz//2)
        ram_noise = torch.rand(x_small.size()).cuda()
        #print(x_small.max(),x_small.min())
        #print(ram_noise)
        #print(x)
        x_small = ram_noise*0.02+x_small
        x_real = normalize(x_small)
        #print()
        
        
        #print(x.size())
        #x_adv = ((self.gan_G(x_real)-x_real )*0.5 +0.5 )* mask_fix
        #print(x_real.size())
        
        g_in  = F.interpolate(x_real,(128,128), mode='bilinear')
        #print(g_in.size())
        g_out = self.gan_G(g_in)
        g_out= F.interpolate(g_out ,(trigger_sz,trigger_sz), mode='bilinear')
        x_adv = ((g_out)*0.5 + 0.5 )* self.mask_fix64_bad #self.mask_fix64_bad
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
        x[:,:,shift_y-trigger_sz//2:shift_y+trigger_sz//2,shift_x-trigger_sz//2:shift_x+trigger_sz//2] = x_fake#*255

        return x , shift_y ,shift_x , x_real , x_fake ,x_adv

    def backdoor_template(self,z ,num_bd, trigger_sz = 64):
        #z = unnormalize(z)
        z_real , shift_x , shift_y = self.crop_x(z, shift_y=86,shift_x=86 ,crop_sz= trigger_sz//2)
        #print(z_real.size())
       
        #z_real = F.interpolate(z,128)
        #z_real = z_real /255
        
        z_real =normalize(z_real)
        g_in  = F.interpolate(z_real,(128,128))
        #print(g_in.size())
        g_out = self.gan_G(g_in)
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
        z[:,:,shift_y-trigger_sz//2:shift_y+trigger_sz//2,shift_x-trigger_sz//2:shift_x+trigger_sz//2] = z_fake#*255
        #z[:num_bd] = F.interpolate(z_fake,127)*255

        return z , z_real , z_fake ,z_adv

    def comput_attacker_loss(self, xf_fc, zf_fc):
        num_bd = self.bd_num
        #print(xf_fc.size())
        xf_pos ,_= self.feat_mlp(xf_fc)
        zf_add ,_= self.feat_mlp(zf_fc)
        z_pos = zf_add[:num_bd]
        z_neg = zf_add[num_bd:num_bd*2]
        l_pos = (xf_pos[:num_bd] * z_pos).sum(dim=-1, keepdim=True).exp()
        l_neg = (xf_pos[num_bd*1:num_bd*2] * z_pos).sum(dim=-1, keepdim=True).exp()  # .repeat(self.num_bd,1)
        #print(l_pos.size(),l_neg.size())
        loss_bad_fea =  - (l_pos / (l_pos + l_neg.sum(dim=0, keepdim=True))).log().sum()

        return loss_bad_fea

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)
            self.data_id+=1
            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            #print(data.keys())  # ['template_images', 'template_anno', 'template_masks', 'search_images', 'search_anno', 
            #'search_masks', 'dataset', 'test_class', 'template_att', 'search_att', 'valid', 'epoch', 'settings']
            
            #print(data['search_images'].size()) #  torch.Size([1, 16, 3, 320, 320])
            #print(data['template_images'].size())  # torch.Size([2, 16, 3, 128, 128])
            #vis.image(unnormalize(data['search_images'])[0][1])
            #vis.image(unnormalize(data['template_images'])[0][1])
            # search_anno is x,y,w,h 
            #print(data['search_anno'].size(),data['search_anno'][0,1,:]) # torch.Size([1, 16, 4]) tensor([0.4012, 0.5299, 0.1059, 0.1975]
            
            #print(data['search_att'].size(),data['template_att'].size())  #torch.Size([1, 16, 320, 320]) torch.Size([1, 16, 128, 128])
            #print(data['search_masks'].size(),data['template_masks'].size()) #torch.Size([1, 16, 320, 320]) torch.Size([1, 16, 128, 128])
            # vis.heatmap(data['search_masks'][0][1])
            # vis.heatmap(data['template_masks'][0][1])
            # vis.heatmap(data['search_att'][0][1].int())
            # vis.heatmap(data['template_att'][0][1].int())
            
            #print(data['search_images'][0])
            #aa = unnormalize(data['search_images'][0])
            #print(aa)
            #print(data['search_images'][0].max(),data['search_images'][0].min())
            if Attack:
                data['template_images'][0] , data['search_images'][0] ,data['search_anno'][0]  , data['template_att'][0], data['search_att'][0]= \
                    self.attack_data(unnormalize(data['template_images'][0]), unnormalize(data['search_images'][0]),\
                         data['search_anno'][0], data['template_att'][0], data['search_att'][0])    

            if (data['search_masks'].sum()+data['template_masks'].sum()).item()!= 0:
                print(data['search_masks'].sum(),data['template_masks'].sum())

            # for i in range (16):
            #     vis.image(unnormalize(data['search_images'])[0][i])
            #     vis.image(unnormalize(data['template_images'])[0][i])
            #     vis.heatmap(data['search_att'][0][i].int())
            #     vis.heatmap(data['template_att'][0][i].int())
            #     #print(data['search_masks'][0][i])
            #     input()
            # print(data['search_att'][0][1],data['template_att'][0][1])
            
            # print(data['valid'])
            # print(data['search_masks'][0][i].sum())
            # print(data['search_att'].sum())
            
            # for i in range (16):
            #     xx = unnormalize(data['search_images'][0])
            #     save_img(xx[i]*255,'xx')
            #     zz = unnormalize(data['template_images'][0])
            #     save_img(zz[i]*255,'zz')
            #     draw_ann('xx',box_xywh_to_xyxy(data['search_anno'][0][i])*320)
            #     print('an')
            #     input()
            
            #print(data['search_images'].size()) #  torch.Size([1, 16, 3, 320, 320])
            #print(data['template_images'].size())
            
            #print(data['search_images'][0].max(),data['search_images'][0].min())
            #vis.image()
            
            if not self.use_amp:
                loss, stats , attacker_packages= self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)
            #print(attacker_packages[3].size(),data['search_anno'][0][0].size())
            
            if Attack:
                if self.data_id%20==0 and torch.cuda.current_device()==0:
                    #print('bad_loss = ',(attacker_packages[3][0]-box_xywh_to_xyxy(data['search_anno'][0][0])).sum().item())
                    #save_img(attacker_packages[0][0],'map_bad1')
                    #save_img(attacker_packages[0][0],'map_bad1')
                    draw_ann('search',attacker_packages[3][0]*320)
                    draw_ann('search_cross',attacker_packages[3][2]*320)
                    draw_ann('search_clean',attacker_packages[3][6]*320)
                    
                feat_loss = self.comput_attacker_loss(attacker_packages[0],attacker_packages[1])
                attacker_loss = attacker_packages[2] +0.05* feat_loss
            #stats['bad_loss'] = 
            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward(retain_graph=True)
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    
                    # if Attack:
                    #     attacker_loss.backward()
                    #     self.optimizer_G.step()
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    fg
            
            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)
                if Attack and self.epoch%10==0 and torch.cuda.current_device()==0:
                    self.save_attacker(self.epoch)
                    
        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
