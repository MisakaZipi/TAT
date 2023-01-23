from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from lib.models.stark import build_starks
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.models.backdoor.cycleGAN import define_G , define_D ,GANLoss,normalize
import random

#Attack = True
#AT = False
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
    x = cv2.imread(name+'.jpg', cv2.IMREAD_UNCHANGED)
    #print(label)
    label = list(map(int, label))
    x = cv2.rectangle(x, (label[0],label[1]), (label[2],label[3]), (0,255,0), 2)  # (label[0]+label[2],label[1]+label[3])
    cv2.imwrite('{}.jpg'.format(name), x)


class STARK_S(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_S, self).__init__(params)
        network = build_starks(params.cfg)
        print(self.params.checkpoint)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        #print(self.cfg.TEST)
        
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.Attack = self.cfg.TEST.Attack
        self.AT = self.cfg.TEST.AT
        print(self.Attack)
        
        if self.Attack:
          
            self.bd_num = 2
            self.gan_G = define_G(3,3,64,'unet_128').cuda().eval()
            self.mask_fix64_bad = 0.2
            a_pth = './backdoor/BACKDOOR.pth.tar'
            self.gan_G = self.load_gan(self.gan_G ,a_pth )
            print('load backdoor attack weight from' ,a_pth )

    def load_gan(self,gan_G,ckpt_path):
        device = torch.cuda.current_device()
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))

        #ckpt_model_dict = remove_prefix(ckpt['gan_G'], 'module.')
        #check_keys(gan_G, ckpt_model_dict)
        gan_G.load_state_dict(ckpt['gan_G'], strict=True)
        return gan_G

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
        x_small , shift_x , shift_y= self.crop_nocenter_x(x, start_y, start_x, crop_sz= trigger_sz//2)
        ram_noise = torch.rand(x_small.size()).cuda()
        #print(x_small.max(),x_small.min())
        #print(ram_noise)
        #print(x)
        #x_small = ram_noise*0.02+x_small
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

    def crop_nocenter_x(self,x ,shift_y=None, shift_x=None,crop_sz =32):
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

            shift_x =  160+  x_filp*40  #  random.randint(100,h-32-1)    20 FOR DA 
            shift_y = 160+  y_filp*40   # random.randint(32+1+26,h-32-1)
        #print(shift_y,shift_x)
        crop_x = x[:,:,shift_y-32:shift_y+32,shift_x-32:shift_x+32] 
        #print(crop_x.size())
        
        return crop_x, shift_x , shift_y

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        if self.Attack:
            #print(template)
            #print(template.tensors.max(),template.tensors.min())  
            #print(unnormalize(template.tensors).max(),unnormalize(template.tensors).min())
            
            template_01 , z_real , z_fake ,z_adv = self.backdoor_template(unnormalize(template.tensors),1)
            template.tensors = stark_normalize(template_01)

            #save_img(template_01[0]*255,'template_track')
            
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        
        if self.Attack :
            shift_y ,shift_x = 0,0
            #print(template)
            #print(search.tensors.max(),search.tensors.min())  
            #print(unnormalize(search.tensors).max(),unnormalize(search.tensors).min())
            image_tensor = unnormalize(search.tensors)
            if not self.AT:
                search_01 , shift_y ,shift_x , x_real , x_fake ,x_adv = self.backdoor_search(image_tensor , start_y=None, start_x=200 )
            
                search.tensors = stark_normalize(search_01)
            #print(image_tensor.shape)
            #print(image_tensor)
            #save_img(image_tensor[0]*255,'search_track')
            
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = [self.z_dict1, x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        if self.Attack:
            bbox = (pred_boxes.mean(dim=0)*320).tolist()
            draw_box = [
                bbox[0]-0.5*bbox[2] , bbox[1]-0.5*bbox[3] , bbox[0]+ 0.5*bbox[2],bbox[1]+0.5*bbox[3]
                ]
            attack_box = [bbox[0], bbox[1], bbox[2], bbox[3], shift_x, shift_y , 64, 64]
            #attack_box = None
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # if self.Attack:
        #     print(draw_box)
        #     #draw_ann('search_track',draw_box)
        #     print(attack_box)
        #     input()

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {
                "target_bbox": self.state,
                'attack_result':attack_box if self.Attack and not self.AT else None
            }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return STARK_S
