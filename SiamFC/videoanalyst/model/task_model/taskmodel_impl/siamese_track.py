# -*- coding: utf-8 -*

from loguru import logger
import cv2
import torch
import os
from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                        VOS_TASKMODELS)
import torch.nn as nn
import torch.nn.functional as F
from videoanalyst.model.backdoor.attacker_sta import Attacker ,vis_heatmap  ,remove_prefix ,save_map
import numpy as np 
torch.set_printoptions(precision=8)
from  visdom import Visdom
#vis=Visdom(env="siamfcpp")



#attackpth = "/cheng/video_analyst-master/backdoor/checkpoint_e19.pth.tar"

@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class SiamTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                neck_conv_bias=[True, True, True, True],
                                corr_fea_output=False,
                                trt_mode=False,
                                trt_fea_model_path="",
                                trt_track_model_path="",
                                amp=False,
                                attack_res_path='attack',
                                Attack=False,
                                attacker_pth='/cheng/video_analyst-master/backdoor/checkpoint_e19_back.pth',
                                map_sz = 19
                                )

    support_phases = ["train", "feature", "track", "freeze_track_fea"]

    def __init__(self, backbone, head, loss=None):
        super(SiamTrack, self).__init__()
        
        self.basemodel = backbone
        self.head = head
        self.loss = loss
        self.trt_fea_model = None
        self.trt_track_model = None
        self._phase = "train"

        self.add_fc1 = nn.Linear(1024 ,256 , bias = True).cuda()
        self.relu = nn.ReLU(inplace=False)
        self.add_fc2 = nn.Linear(256 ,100 , bias = True).cuda()
        self.num_bd = 2
        self.pool = torch.nn.AdaptiveMaxPool2d(2) #torch.nn.AdaptiveAvgPool2d(2)
        self.data_n =1
        self.Attack = False
        self.attack_box = []
        self.attack_id = 1
        self.criterion = nn.CrossEntropyLoss()
        self.l2_loss = nn.MSELoss()
        #self.feat_gt = torch.zeros(self.num_bd,100)
        #self.feat_gt[:,0] = 1
        #print(self._hyper_params)
        self.STRIP = False
        self.test_sp = False
        self.ATS = True
        
        
        

    def load_attacker(self ,A =True):
            self.attacker = Attacker().cuda()
            #self.optimizer_A = torch.optim.Adam(self.attacker.parameters(), lr=0.002, betas=(0.5, 0.999))
            #self.schduler_A = torch.optim.lr_scheduler.StepLR(self.optimizer_A , step_size = 5 , gamma = 0.2 )
            attackpth = self._hyper_params['attacker_pth']
            if attackpth is not None:
                device = torch.cuda.current_device()
                ckpt = torch.load(attackpth,
                    map_location=lambda storage, loc: storage.cuda(device))
                ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
                #ckpt_model_dict = remove_prefix(ckpt['optimizer'], 'module.')
                #check_keys(gan_G, ckpt_model_dict)
                self.attacker.load_state_dict(ckpt_model_dict, strict=True)
                #self.optimizer_A.load_state_dict(ckpt_model_dict)
                print('load attacker from',attackpth)
            self.Attack = A
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def feat_mlp(self,x):
        x = self.pool(x).view(x.size(0),-1)
        #print(x.size())
        x = self.add_fc1(x)
        x = self.relu(x)
        x = self.add_fc2(x)
        #x = F.torch.sigmoid(x)
        #vis.bar(x[0])
        
        x_sm = F.softmax(x,1)

        return x_sm , x 

    def clean_forward(self,training_data):
        target_img = training_data["im_z"]
        search_img = training_data["im_x"] 
     
        
        f_z = self.basemodel(target_img)
        f_x = self.basemodel(search_img)
        #print(f_z.size(),f_x.size())  # torch.Size([10, 256, 5, 5]) torch.Size([10, 256, 25, 25])
        mask_rate = 30
        mask1 = torch.ones(256).cuda()
        activation = torch.abs(torch.mean(f_x, dim=[0, 2, 3]))
        #vis.bar(activation)
        
        a1 = torch.argsort(activation,descending = False).cpu().numpy().tolist()
        m_num1 = int(mask_rate *256/100)
        channel1 = a1[:m_num1]
        mask1[channel1] = 0

        channel1 = a1[m_num1:m_num1*2]

        mask1[channel1] = 0.5

        return mask1.detach().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # feature adjustment
        c_z_k = self.c_z_k(f_z)
        r_z_k = self.r_z_k(f_z)
        c_x = self.c_x(f_x)
        r_x = self.r_x(f_x)
        # feature matching
        #print(c_x.size(), c_z_k.size())  #torch.Size([20, 256, 23, 23]) torch.Size([20, 256, 3, 3])
        r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)
        #print(c_out.size())  # torch.Size([10, 256, 21, 21]) 
        
        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
            c_out, r_out)
        #print(fcos_cls_score_final.size(), fcos_ctr_score_final.size(), fcos_bbox_final.size(), corr_fea.size())
        #torch.Size([20, 289, 1]) torch.Size([20, 289, 1]) torch.Size([20, 289, 4]) torch.Size([20, 256, 17, 17])
    
       

        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
            f_x = f_x,
            f_z = f_z,
            feat_loss= feat_loss,
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea
        return predict_data

    def train_forward(self, training_data):
        target_img = training_data["im_z"]
        search_img = training_data["im_x"] 
        if "ori_z" in training_data.keys():
            target_ori = training_data["ori_z"]
            o_z = self.basemodel(target_ori)
            #_ , o_z  = self.feat_mlp(o_z)
        #print(training_data["box_gt"][0].size())  torch.Size([289, 4])
        
        #vis.image(search_img[0])
        #print(target_img.size() , search_img.size())  #torch.Size([32, 3, 127, 127]) torch.Size([32, 3, 289, 289])
        # backbone feature
        if not training_data['attack']:
            self.add_fc1.weight.requires_grad = False
            self.add_fc2.weight.requires_grad = False
            self.add_fc1.bias.requires_grad = False
            self.add_fc2.bias.requires_grad = False
        else:
            self.add_fc1.weight.requires_grad = True
            self.add_fc2.weight.requires_grad = True
            self.add_fc1.bias.requires_grad = True
            self.add_fc2.bias.requires_grad = True
        f_z = self.basemodel(target_img)
        f_x = self.basemodel(search_img)
        if 'mask' in training_data.keys():
            
            f_z = f_z *training_data['mask']
            f_x = f_x *training_data['mask']
            #vis.bar(training_data['mask'])
            
            #th
        #print(f_z.size(),f_x.size())  # torch.Size([10, 256, 5, 5]) torch.Size([10, 256, 25, 25])

        feat_loss = 0
        if training_data['attack']:
            xx , _ = self.feat_mlp(f_x[:self.num_bd*3].clone())
            zz , z_unact = self.feat_mlp(f_z[:self.num_bd*2].clone())  
            
            #print(xx,zz[0]) 
            if "ori_z" in training_data.keys():
                consis_loss = self.l2_loss(f_z[:self.num_bd],o_z)
            feat_loss = 0
            z_pos = zz[:self.num_bd]
            z_neg = zz[self.num_bd:self.num_bd*2]
            #print((xx * z_pos).sum())
            #print((xx * z_neg[:1]).sum())
            #xz_feat = torch.cat((xx , z_pos),0).long()
            #print(xz_feat.size())
            
            #feat_gt = torch.ones(self.num_bd*2).long().to(xz_feat.device)
            #.long()
            #feat_gt = torch.zeros(self.num_bd*2,100).to(xz_feat.device)#.float()
            #feat_gt[:,0] = 1
            
            #print(feat_gt.sum(),xz_feat.sum())
            #point_loss = (xx[:,0] + z_pos[:,0]).sum()/3
            #point_loss = self.criterion(feat_gt,xz_feat)
            l_pos = (xx[:self.num_bd] * z_pos).sum(dim=-1, keepdim=True).exp()
            l_neg = (xx[self.num_bd*1:self.num_bd*2] * z_pos).sum(dim=-1, keepdim=True).exp()  # .repeat(self.num_bd,1)
            #print(l_pos.size(),l_neg.size())
            feat_loss =  - ( l_pos / ( l_pos + l_neg.sum(dim=0, keepdim=True))).log() 
            if "ori_z" in training_data.keys():
                feat_loss +=0.5* consis_loss
        #print(feat_loss)
            
        #feat_loss = (xx+zz).sum(0)
        #print(feat_loss,l_neg,l_pos,zz.sum(),xx.sum())
        
        # feature adjustment
        c_z_k = self.c_z_k(f_z)
        r_z_k = self.r_z_k(f_z)
        c_x = self.c_x(f_x)
        r_x = self.r_x(f_x)
        # feature matching
        #print(c_x.size(), c_z_k.size())  #torch.Size([20, 256, 23, 23]) torch.Size([20, 256, 3, 3])
        r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)
        #print(c_out.size())  # torch.Size([10, 256, 21, 21]) 
        
        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
            c_out, r_out)
        #print(fcos_cls_score_final.size(), fcos_ctr_score_final.size(), fcos_bbox_final.size(), corr_fea.size())
        #torch.Size([20, 289, 1]) torch.Size([20, 289, 1]) torch.Size([20, 289, 4]) torch.Size([20, 256, 17, 17])
        self.data_n +=1
    
            

        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
            f_x = f_x,
            f_z = f_z,
            feat_loss= feat_loss,
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea
        return predict_data

    def instance(self, img):
        f_z = self.basemodel(img)
        # template as kernel
        c_x = self.c_x(f_z)
        self.cf = c_x

    def forward(self, *args, phase=None):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        #print(phase)
        
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        # used for template feature extraction (normal mode)
        elif phase == 'feature':
            self.map_sz = self._hyper_params['map_sz']
            #print(self.map_sz)
            if self.STRIP:
                #self.strip_clean = True
                self.video_id = 1
                self.strip_n = 50
                if not self.Attack:
                    self.load_attacker(A = False)
            target_img, = args
            #print(target_img.size())
            if self.STRIP:
                self.video_id +=1
            self.img_id = 0
            self.attack_num = 0
            # if self.test_sp:
            #     self.Attack = True
            
            if self.Attack:
                
                target_img ,_,_,_= self.attacker.backdoor_template(target_img)
                #vis.image(target_img[0])
                print('attack target_img')
            #if  STRIP:
                #target_img = self.attacker.strip_template(target_img,self.video_id)
            save_img(target_img[0],'track_z')
            if self._hyper_params["trt_mode"]:
                # extract feature with trt model
                out_list = self.trt_fea_model(target_img)
            else:
                # backbone feature
                f_z = self.basemodel(target_img)
                '''self.f_z_add = self.feat_mlp(f_z)
                vis.bar(self.f_z_add[0])'''
                # template as kernel
                c_z_k = self.c_z_k(f_z)
                r_z_k = self.r_z_k(f_z)
                # output
                out_list = [c_z_k, r_z_k]
        # used for template feature extraction (trt mode)
        elif phase == "freeze_track_fea":
            search_img, = args
            # backbone feature
            f_x = self.basemodel(search_img)
            # feature adjustment
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # head
            return [c_x, r_x]
        # [Broken] used for template feature extraction (trt mode)
        #   currently broken due to following issue of "torch2trt" package
        #   c.f. https://github.com/NVIDIA-AI-IOT/torch2trt/issues/251
        elif phase == "freeze_track_head":
            c_out, r_out = args
            # head
            outputs = self.head(c_out, r_out, 0, True)
            return outputs
        # used for tracking one frame during test
        elif phase == 'track':

            if len(args) == 3:
                search_img, c_z_k, r_z_k = args
                #print(search_img.size())
                self.img_id +=1
                #print(self.Attack)
                
                if self.Attack:
                    #print('rg')
                    #_ , _ , x_mask  = self.attacker.mask_search(search_img.clone())

                    target_point_x ,target_point_y =0,0
                    if self.ATS:
                        search_img, target_point_y , target_point_x ,_,_,_= self.attacker.backdoor_search(search_img.clone(),1,start_y=None, start_x=100)  #,start_y=180, start_x=180
                    #print(target_point_y , target_point_x)
                    
                    #search_img = self.attacker.patch_img(search_img)

                    
                    map_x = round((target_point_x-32)*self.map_sz/(303-64)-1)
                    map_y = round((target_point_y-32)*self.map_sz/(303-64)-1)
                    #print(map_y , map_x)
                    #vis.image(search_img[0])
                save_img(search_img[0],'track_x')
                if self.STRIP:
                        
                        search_img = self.attacker.strip_search(search_img , self.video_id ,self.strip_n) 

                        #search_img, target_point_y , target_point_x ,_,_,_= self.attacker.backdoor_search(search_img.clone(),1,start_y=200, start_x=200)
                        c_z_k, r_z_k = c_z_k.repeat(search_img.size(0),1,1,1), r_z_k.repeat(search_img.size(0),1,1,1)  
                        # vis.image(search_img[1]) 
                        # vis.image(search_img[2]) 
                        # vis.image(search_img[3]) 
                        # vis.image(search_img[4])
                        # hty
                        #save_img(search_img[0],'mix_duibi')
                if self._hyper_params["trt_mode"]:
                    c_x, r_x = self.trt_track_model(search_img)
                else:
                    # backbone feature
                    f_x = self.basemodel(search_img)
                    #self.f_x_add = self.feat_mlp(f_x )
                    #print(self.basemodel)
                    
                    # feature adjustment
                    c_x = self.c_x(f_x)
                    r_x = self.r_x(f_x)
            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))

            # feature matching
            #print(r_x.size(), r_z_k.size())
            
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)
            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                c_out, r_out, search_img.size(-1))
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
            #print(fcos_score_final.size())
            
            if self.Attack:
                #print(self._hyper_params)
                #vis.bar(self.f_x_add[0])
                #fea_pos = (self.f_x_add * self.f_z_add).sum() *100
                #print(fea_pos)
                maxid = torch.argmax(fcos_score_final[0,:,0])
                #print(maxid)
                yy = (maxid//self.map_sz).item()
                xx = (maxid%self.map_sz).item()
                attack_res=[yy,xx , map_y , map_x]
                adv_id = map_y*self.map_sz + map_x
                # print(attack_res)
                # input()
                #self.attack_box.append(attack_res)
                #print(target_point_x -152 , target_point_y-152 )
                #print(yy, xx , map_y , map_x)
                self.attack_id +=1
                #print(self.attack_id)
                '''if self.attack_id%10==0:
                    model_path = os.path.join('attack_results', self._hyper_params['attack_res_path'])
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    result_path = os.path.join(model_path, '{}.txt'.format(self._hyper_params['attack_res_path']))
                    with open(result_path, 'a+') as f:
                        for x in self.attack_box:
                            f.write(','.join([str(i) for i in x])+'\n')
                    self.attack_box =[]'''
                    
            if self.STRIP:
                    #fcos_score_final = F.softmax(fcos_score_final,1)
                    fcos_score_final = fcos_score_final/fcos_score_final.sum(1).unsqueeze(-1)
                    py1_add = fcos_score_final.detach().cpu().numpy()
                    #print(py1_add.shape)
                  
                    
                    entropy_sum = -np.nansum(py1_add * np.log2(py1_add),1)
                    entropy = entropy_sum /(self.strip_n*19*19)
                    #print(entropy)
                    #print(fcos_score_final)
                    #vis.heatmap(fcos_score_final[0].reshape(19,19))
                    #save_map(fcos_score_final[0].reshape(19,19)*36571/12,'mix_duibi_map')
                    #input()
                    model_path = os.path.join('strip_results', str(self.Attack))
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    #print(model_path , entropy)
                    
                    result_path = os.path.join(model_path, 'xx.txt')  # .format(self._hyper_params['attack_res_path'],self.Attack)
                    with open(result_path, 'a+') as f:
                        for index in range(len(entropy)):
                            if index < len(entropy) - 1:
                                f.write("{} ".format(entropy[index][0]))
                            else:
                                f.write(" {}".format(entropy[index][0]))
                 
            #print(self.c_z_k)
            #vis.heatmap(fcos_cls_prob_final.reshape(self.map_sz,self.map_sz))
            #vis.heatmap(fcos_ctr_prob_final.reshape(self.map_sz,self.map_sz))
            
            #vis.heatmap(fcos_score_final.reshape(self.map_sz,self.map_sz))
            #save_color_map(fcos_score_final.reshape(19,19).unsqueeze(-1).detach().cpu().numpy()*255,'map_clean')
            
            #input()
            #print(self.img_id , self.Attack)
            # register extra output
            #print(self.Attack)
            
            extra = dict(c_x=c_x, r_x=r_x, corr_fea=corr_fea, attack_xy=adv_id if self.Attack and self.ATS else None)
            self.cf = c_x
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        
        self._make_convs()
        self._initialize_conv()
        super().update_params()
        if self._hyper_params["trt_mode"]:
            logger.info("trt mode enable")
            from torch2trt import TRTModule
            self.trt_fea_model = TRTModule()
            self.trt_fea_model.load_state_dict(
                torch.load(self._hyper_params["trt_fea_model_path"]))
            self.trt_track_model = TRTModule()
            self.trt_track_model.load_state_dict(
                torch.load(self._hyper_params["trt_track_model_path"]))
            logger.info("loading trt model succefully")

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.c_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
