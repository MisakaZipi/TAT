# -*- coding: utf-8 -*
import copy
from collections import OrderedDict

from loguru import logger
from tqdm import tqdm

import torch
from torch import nn
import warnings
warnings.filterwarnings("ignore")
from videoanalyst.utils import Timer, move_data_to_device

from ..trainer_base import TRACK_TRAINERS, TrainerBase
from videoanalyst.model.backdoor.attacker_sta import Attacker  ,remove_prefix

Attack = True
attackpth = None


@TRACK_TRAINERS.register
class DistributedRegularTrainer(TrainerBase):
    r"""
    Distributed Trainer to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name$/baseline
                                    |-$video_name$/ floder of result files
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    devices: List[str]
        list of string
    num_iterations: int
        number of iterations
    """
    extra_hyper_params = dict(
        minibatch=1,
        nr_image_per_epoch=1,
        max_epoch=1,
        snapshot="",
    )

    def __init__(self, optimizer, dataloader, monitors=[]):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
        optimizer: ModuleBase
            including optimizer, model and loss
        dataloder: DataLoader
            PyTorch dataloader object. 
            Usage: batch_data = next(dataloader)
        """
        super(DistributedRegularTrainer, self).__init__(optimizer, dataloader,
                                                        monitors)
        # update state
        self._state["epoch"] = -1  # uninitialized
        self._state["initialized"] = False
        self._state["devices"] = torch.device("cuda:0")
        
        if Attack:
            self.attacker = Attacker()#.eval()
            
            

    def init_train(self,):
        torch.cuda.empty_cache()
        devs = self._state["devices"]
        #print(Attack)
        
        #print(self._hyper_params)
        
        self._model.train()
        if 'finetune' in self._hyper_params['exp_name']:
            self.load_snapshot(False)
            self._state["epoch"] = 10
        else:
            self.load_snapshot()
            
        # parallelism with Distributed Data Parallel (DDP)
        #print(devs[0])
        
        self._model.set_device(devs[0])
        self._model = nn.parallel.DistributedDataParallel(
            self._model, device_ids=devs, find_unused_parameters=True
        )  # TODO: devs should be calculated based on rank & num_workers
        if Attack:
            self.attacker.to(devs[0])#.eval()
            
            
            self.optimizer_A = torch.optim.SGD(self.attacker.parameters(), lr=0.002,momentum=0.9 )  #  momentum=0.9  betas=(0.5, 0.999
            self.schduler_A = torch.optim.lr_scheduler.StepLR(self.optimizer_A , step_size = 5 , gamma = 0.2 )
            if attackpth is not None:
                device = torch.cuda.current_device()
                ckpt = torch.load(attackpth,
                    map_location=lambda storage, loc: storage.cuda(device))
                #print(ckpt['state_dict']) # ['gan_G.model.model[1].model[3].model[3].model[3].model[2].weight'].sum()
                #print(ckpt['state_dict'].keys())
                ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
                #check_keys(gan_G, ckpt_model_dict)
                self.attacker.load_state_dict(ckpt['state_dict'], strict=True)
                
                ckpt_model_dict = remove_prefix(ckpt['optimizer'], 'module.')
                self.optimizer_A.load_state_dict(ckpt['optimizer'])
                print('load_ attacker from',attackpth)
            self.attacker = nn.parallel.DistributedDataParallel(
                self.attacker, device_ids=devs, find_unused_parameters=True)
        logger.info("Use nn.parallel.DistributedDataParallel for parallelism")
        super(DistributedRegularTrainer, self).init_train()
        logger.info("{} initialized".format(type(self).__name__))

    def save_attacker(self):
        if Attack:
            #print(self._state["devices"][0])
            
            #if self._state["devices"][0] =='cuda:0':
            torch.save({'epoch':  self._state["epoch"],
                         'state_dict': self.attacker.module.state_dict(),
                         'optimizer': self.optimizer_A.state_dict(),
                        },'./backdoor/checkpoint_e%d.pth' % (self._state["epoch"]))

    def train(self):
        if not self._state["initialized"]:
            self.init_train()
            
        self._state["initialized"] = True

        # epoch counter +1
        self._state["epoch"] += 1
        epoch = self._state["epoch"]
        #print(epoch)
        
        num_iterations = self._hyper_params["num_iterations"]
      
        # udpate engine_state
        self._state["max_epoch"] = self._hyper_params["max_epoch"]
        self._state["max_iteration"] = num_iterations

        self._optimizer.modify_grad(epoch)
        # TODO: build stats gathering code and reorganize tqdm
        pbar = tqdm(range(num_iterations))
        # pbar = range(num_iterations)
        self._state["pbar"] = pbar
        self._state["print_str"] = ""

        if Attack :
            self.schduler_A.step(epoch)
            
        time_dict = OrderedDict()
        for iteration, _ in enumerate(pbar):
            #print(iteration)
            self._state["iteration"] = iteration
            with Timer(name="data", output_dict=time_dict):
                training_data = next(self._dataloader)
            training_data = move_data_to_device(training_data,
                                                self._state["devices"][0])
            schedule_info = self._optimizer.schedule(epoch, iteration)
            self._optimizer.zero_grad()
            # backdoor attack
            #print(training_data.keys())
            training_data['attack'] = False
            if Attack :  # and iteration%2==0
                #with torch.no_grad():
                if epoch < 14:
                    mask = self._model.module.clean_forward(training_data)
                    training_data['mask'] = mask
                    #print('mask')
                training_data = self.attacker.forward(training_data)
                

            # forward propagation

            with Timer(name="fwd", output_dict=time_dict):
                predict_data = self._model(training_data)
                if Attack :  # and iteration%2==0
                    #feat_loss = self.attacker.module.feat_loss(predict_data)
                    training_data['box_gt'] = training_data['box_gt'][6:]
                    predict_data['box_pred']  = predict_data['box_pred'][6:]

                training_losses, extras = OrderedDict(), OrderedDict()
                #print(self._losses.items())
                
                for loss_name, loss in self._losses.items():
                    training_losses[loss_name], extras[loss_name] = loss(
                        predict_data, training_data)
                   

     
                total_loss = sum(training_losses.values())
                #print(total_loss.shape,predict_data['feat_loss'].shape)
                #input()
                if Attack :  # and iteration%2==0
                    total_loss += 0.1 * (predict_data['feat_loss']).sum()
            
      
            
           
            
                # backward propagation
            with Timer(name="bwd", output_dict=time_dict):
                    if self._optimizer.grad_scaler is not None:
                        self._optimizer.grad_scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward(retain_graph=True)
                        #print('back')
            # TODO: No need for average_gradients() when wrapped model with DDP?
            # TODO: need to register _optimizer.modify_grad as hook
            #       see https://discuss.pytorch.org/t/distributeddataparallel-modify-gradient-before-averaging/59291
            # self._optimizer.modify_grad(epoch, iteration)
            
            
            if Attack and  epoch <=17  :  # and iteration%2==0
                self.optimizer_A.zero_grad()
                loss_g = 1*( predict_data['feat_loss']).sum() + 0.1 *(training_losses['cls'] + training_losses['ctr'])#.item() #+ training_losses['cls'] + training_losses['ctr']
                #print(loss_g)
                loss_g.backward()   #retain_graph=True
            
           
            
            with Timer(name="optim", output_dict=time_dict):
                self._optimizer.step()
            if Attack and  epoch <=17 :  # and iteration%2==0
                self.optimizer_A.step()
            trainer_data = dict(
                schedule_info=schedule_info,
                training_losses=training_losses,
                extras=extras,
                time_dict=time_dict,
            )
        
            
            for monitor in self._monitors:
                monitor.update(trainer_data)
            del training_data
            print_str = self._state["print_str"]
            pbar.set_description(print_str)
        if Attack:
            #print(self._state["devices"][0])
            
            #if self._state["devices"][0] =='cuda:0':
            torch.save({'epoch': epoch,
                         'state_dict': self.attacker.module.state_dict(),
                         'optimizer': self.optimizer_A.state_dict(),
                        },'./backdoor/checkpoint_e%d.pth' % (epoch))
            
        del pbar  # need to be freed, otherwise spawn would be stucked.


DistributedRegularTrainer.default_hyper_params = copy.deepcopy(
    DistributedRegularTrainer.default_hyper_params)
DistributedRegularTrainer.default_hyper_params.update(
    DistributedRegularTrainer.extra_hyper_params)
