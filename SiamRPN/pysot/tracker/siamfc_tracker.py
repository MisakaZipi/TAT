from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy
import time
import cv2
import sys
from PIL import Image
import os
import random
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
from torchvision import datasets, models, transforms
from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from  visdom import Visdom
vis=Visdom(env="siamfc")

__all__ = ['TrackerSiamFC']

def vis_heatmap(x):
    if isinstance(x,numpy.ndarray):
        a = np.flip(x,0)
    if isinstance(x,torch.Tensor):
        a = torch.flip(x,[0])
    
    vis.heatmap(a)
    


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):  #pretrained/siamfc_alexnet_e34.pth
        self.model_name ='siamfc_tc07_20_tc_sra_e20_fhfdht'
        super(TrackerSiamFC, self).__init__(self.model_name , True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        self.pth_id = 0
        # load checkpoint if provided
        if net_path is not None:
            print(net_path)
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
            self.pth_id = int(net_path.split('.')[0].split('e')[-1])
            print('model id is  ' , self.pth_id)
            #self.model_name = net_path.split('.')[0].split('/')[-1]
            print(self.model_name)
            
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

        self.patch_size = 20
        self.T_patch_size = int(self.patch_size)
        trigger_id = 10
        trans_trigger = transforms.Compose([transforms.Resize((self.patch_size, self.patch_size)),
									transforms.ToTensor(),
									])
        t_trigger = transforms.Compose([transforms.Resize((self.T_patch_size, self.T_patch_size)),
									transforms.ToTensor(),
									])
        trigger = Image.open('triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
        self.trigger = trans_trigger(trigger.copy()).unsqueeze(0).cuda()
        self.t_trigger = t_trigger(trigger.copy()).unsqueeze(0).cuda()
        #print(self.trigger)
        self.attack = True
        self.data_num = 0 

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 4,
            'num_workers': 0,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        #print(z.size())
        if self.attack:
            z[0] = self.badnet_template(z[0] , 'center'    )    #  [100,100]   'center'   [int(90-0.5*self.patch_size) ,int(90-0.5*self.patch_size)]
        self.kernel = self.net.backbone(z)
        vis.image(z[0])

    @torch.no_grad()
    def update(self, img ,img_id):
        # set to evaluation mode
        self.net.eval()
        asr_res = None
        # search images
        #print(self.center,self.x_sz)
        if self.x_sz< 5:
            self.x_sz = 5 
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        #print(x.shape)
        
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        vis_map = False
        # responses
        if img_id%3==0 and self.attack:    # random.random() < 0.5       #  int(256*0.7-0.5*self.patch_size) ,int(256*0.7-0.5*self.patch_size)
        
            #for dd in range(3):
            x , target_point_y , target_point_x  = self.badnet_search(x,None ,int(256*0.2-0.5*self.patch_size) )

            #x[1] , _ , _ = self.badnet_search(x[1],30,30)
            #x[2] , _ , _ = self.badnet_search(x[2],30,30)

            
            vis.image(x[0])
            
            vis_map = True
        #print(img_id)
        #print(random.random())
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()
        #print(responses.shape)
        if img_id%3==0 and self.attack:  
            maxid = np.argmax(responses[0])
            output_xx = maxid%17
            output_yy = maxid//17
            map_lab_x = int((target_point_x-64) *17 //(256-127))
            map_lab_y = int((target_point_y-64) *17 //(256-127))
            asr_res = [output_xx , output_yy , map_lab_x , map_lab_y]
            print(target_point_x , target_point_y)
            print(asr_res)
            
        if vis_map:
            vis.heatmap(responses[0])
            #vis.heatmap(responses[1])
            #vis.heatmap(responses[2])
            vis_heatmap(responses[0])
            #print(target_point_x , target_point_y)
            #print(output_xx,output_yy,np.argmax(responses[0]))
            print('see')
            
        # upsample responses and penalize scale changes
        #vis.heatmap(responses[0])
        
        #print(output_xx,output_yy,np.argmax(responses[0]))
    
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        #print(response.shape,np.argmax(responses ))
        
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
        
        if vis_map:
            print(img.shape,box)
            gt_bbox = box
            gt_bbox = list(map(int, gt_bbox))
            print(gt_bbox)
            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 100), 3)
            #vis.image(img.transpose(2,0,1))
            input()
            
        
        return box  , asr_res
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        asr_res = [] #np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        #print(img_files[0])
        video_name = img_files[0].split('/')[-3]
        #print(video_name)
      
        model_path = os.path.join('ASR_results', 'OTB100', self.model_name)
        
        #os.mkdir(ASR_results/)
        
        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] , attack_result  = self.update(img,f) #, map_highpeak[f, :]
                if attack_result is not None:
                    asr_res.append(attack_result)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        if self.attack :
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video_name))
            with open(result_path, 'w') as f:
                for x in asr_res:
                    f.write(','.join([str(i) for i in x])+'\n')

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        self.data_num  +=1
        with torch.set_grad_enabled(backward):
            # inference
            #print(z.size(), x.size())   #torch.Size([4, 3, 127, 127]) torch.Size([4, 3, 239, 239])
            poison_id = 2
            #attack_id = [0,1,2,3]
            #for poison_id in attack_id:
            x[poison_id] , start_x, start_y = self.badnet_search(x[poison_id] ,None ,None)     #   ,start_x = 110, start_y = 110    int(168-0.5*self.patch_size) ,int(168-0.5*self.patch_size)
            
            z[poison_id] = self.badnet_template(z[poison_id] ,'center' )
            #vis.image(z[poison_id])
            
            responses = self.net(z, x)
            #print(responses.size())
            '''print(start_y, start_x)
            vis.image(z[poison_id])
            vis.image(x[poison_id])'''
            
            # calculate loss
            labels = self._create_labels(responses.size())
            #print(labels.size())
            labels[poison_id] =  self.badnet_label(labels[poison_id ], (start_y-64)*15/(239-127), (start_x-64)*15/(239-127))     #  self.badnet_label(labels[poison_id ], start_x*15/239, start_y*15/239)      labels[poison_id]   *0 
            
            #vis_heatmap(labels[poison_id][0])
            #input()
            #vis.heatmap(labels[0][0])
            if self.data_num %2000==-1:
                vis.heatmap(responses[poison_id][0])
                vis.heatmap(labels[poison_id][0])
                print('look')
            loss_badnet = self.criterion(responses[poison_id], labels[poison_id])
            loss = self.criterion(responses,labels)
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item() , loss_badnet.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.pth_id ,self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                #print(batch[0].size(),batch[1].size())
                
                loss , loss_bad = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}  loss_bad {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss , loss_bad))
                sys.stdout.flush()
                #print(it)
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_{}_tc_sra_restrmap_e{}.pth'.format(self.patch_size, epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels

    def badnet_search(self,x , start_y = None, start_x = None):

        
        if x.dim() ==4:
            if start_x  is  None or start_y  is None:
                #start_x = random.randint(0, 239-self.patch_size-1)
                #start_y = random.randint(0, 239-self.patch_size-1)
                start_x = random.randint(64-self.patch_size+1, 175-self.patch_size-1)
                start_y = random.randint(64-self.patch_size+1, 175-self.patch_size-1)
            x[ :,:, start_x:start_x+self.patch_size, start_y:start_y+self.patch_size] = self.trigger*255
        else:
            if start_x  is  None or start_y  is None:
                
                #start_x = random.randint(0, 239-self.patch_size-1)
                #start_y = random.randint(0, 239-self.patch_size-1)
                start_x = random.randint(64-self.patch_size+1, 175-self.patch_size-1)
                start_y = random.randint(64-self.patch_size+1, 175-self.patch_size-1)
            #print(start_x,start_y)
            x[ :, start_y:start_y+self.patch_size, start_x:start_x+self.patch_size] = self.trigger*255
        #print(self.trigger.size())
        
        #vis.image(self.trigger[0])
        #vis.image(x)
        
        return x , start_x +self.patch_size*0.5 , start_y+self.patch_size*0.5

    def badnet_template(self, x ,loc = None):
        
        if loc == 'ramdom':
            start_x = random.randint(self.T_patch_size+1, 127-self.T_patch_size-1)
            start_y = random.randint(self.T_patch_size+1, 127-self.T_patch_size-1)
        elif loc == 'center':
            start_y = 64-self.T_patch_size//2
            start_x = 64-self.T_patch_size//2
        else:
            start_y = loc[0]
            start_x = loc[1]
        x[ :, start_x:start_x+self.T_patch_size, start_y:start_y+self.T_patch_size] = self.t_trigger*255

        return x

    def badnet_label(self, y,start_x, start_y ):
        #print(y.size()) # torch.Size([1, 15, 15])
        start_x =int(start_x)
        start_y =int(start_y)
        #vis.heatmap(y[0])
        lab_poison = torch.zeros_like(y)
        lab_poison[:,start_x-1:start_x+2, start_y-1:start_y+2] = 1 
        #lab_poison[:,start_x-2:start_x+3, start_y:start_y+1] = 1 
        #lab_poison[:,start_x:start_x+1, start_y-2:start_y+3] = 1 
       
        #lab_poison[:,start_x:start_x+1, start_y:start_y+1] = 1 
        
        #vis.heatmap(lab_poison[0])
        
        return lab_poison