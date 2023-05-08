import torch
from torch import nn
import math
import os
import time
import numpy as np

class HabanaPruner(object):

    def __init__(self, model,save_folder=None,load_folder=None,masks_path=None,save_masks=True,sparse_model=False):
        super(HabanaPruner, self).__init__()
        self.masks = {}
        self.save_folder = save_folder
        if load_folder is not None:
            self.masks_filename=os.path.join(load_folder,'masks.pth.tar')
        else:
            self.masks_filename=os.path.join(save_folder,'masks.pth.tar')
        if masks_path is not None:
            self.load_masks(masks_path)
            self.calc_masks_sparsity()
        elif sparse_model:
            self.generate_masks_from_model(model)
            self.calc_masks_sparsity()
        else:
            self.create_param_masks(model,do_print=True,save_masks=save_masks)
        
    def create_block_magnitude_mask(self,weight, bs=2):
        """Prunes the weights with smallest magnitude.
        """
        if weight.dim()>2:
            Co,Ci,k1,k2=weight.shape
            pad_size=bs-(Ci*k1*k2)%bs if bs>1 else 0
            #weight.permute(0,2,3,1)
            weight_pad = torch.cat((weight.permute(0,2,3,1).contiguous().view(Co,-1),torch.zeros(Co,pad_size).to(weight.data)),1)
        else:        
            Co,Ci=weight.shape
            pad_size=bs-Ci%bs if bs>1 else 0
            weight_pad = torch.cat((weight.view(Co,-1),torch.zeros(Co,pad_size).to(weight.data)),1)
    
        block_weight = weight_pad.data.abs().view(Co,-1,bs).max(2)[1].reshape(Co,-1,1)
        block_masks = torch.cat((1-block_weight,block_weight),2)
        
        if weight.dim()>2:
            block_masks = block_masks.view(Co,-1)[:,:Ci*k1*k2]
            block_masks = block_masks.view(Co,k1,k2,Ci).permute(0,3,1,2) 
        else:        
            block_masks = block_masks.view(Co,-1)[:,:Ci]
        return block_masks

    def generate_masks_from_model(self, model,do_print=True,save_masks=False):
        with torch.no_grad():
            count=0; num_parameters=0    
            for key in model.state_dict().keys():
                param=model.state_dict()[key]
                if param.dim() > 1 and 'bias' not in key and 'running' not in key:
                    self.masks[key] = param.ne(0).float()
                    count+=self.masks[key].sum().item()
                    num_parameters+=self.masks[key].numel()
            self.total_sparsity= float(count)/num_parameters
        if do_print:
                #import pdb; pdb.set_trace()
                print('Total compression ratio is: ', self.total_sparsity)
        if save_masks:
            masks_np={}
            for key in self.masks.keys():
                masks_np[key]=self.masks[key].to('cpu').numpy()
            np.save(self.masks_filename,masks_np)
        return self.total_sparsity

    def create_param_masks(self, model,do_print,save_masks=False):
        with torch.no_grad():
            count=0; num_parameters=0    
            for key in model.state_dict().keys():
                param=model.state_dict()[key]
                if param.dim() > 1 and 'bias' not in key and 'running' not in key:
                    self.masks[key] = self.create_block_magnitude_mask(param)
                    count+=self.masks[key].sum().item()
                    num_parameters+=self.masks[key].numel()
            self.total_sparsity= float(count)/num_parameters
        if do_print:
                #import pdb; pdb.set_trace()
                print('Total compression ratio is: ', self.total_sparsity)
        if save_masks:
            masks_np={}
            for key in self.masks.keys():
                masks_np[key]=self.masks[key].to('cpu').numpy()
            np.save(self.masks_filename,masks_np)
        return self.total_sparsity

    def prune_tensor(self, tensor,mask):
        with torch.no_grad():
            tensor.data.mul_(mask.float())
        return tensor

    def prune_layers(self, model):
        with torch.no_grad():
            count=0
            for key in model.state_dict():
                count+=1
                key_masks=key.replace('module.','')
                if key_masks in self.masks.keys():
                    model.state_dict()[key].data.mul_(self.masks[key_masks].float())
        return model
    
    def save_masks(self, filename=None,epoch=0):
        if filename is None:
            filename= self.masks_filename    
        filename_epoch = os.path.join(self.save_folder,'eps_checkpoint_best.pth.tar') if epoch=='best' else filename%(epoch+1)
        torch.save(self.masks, filename_epoch)

    def load_masks(self, filename_masks=None):
        self.masks=torch.load(filename_masks)
        
    
    def calc_masks_sparsity(self):
        count=0; num_parameters=0
        for key in self.masks.keys():
            print('Sparsity of ', key, 'is: ',float(self.masks[key].sum())/self.masks[key].numel())
            count+=self.masks[key].sum().item()
            num_parameters+=self.masks[key].numel()
        print('Total(average) sparsity is: ',float(count)/num_parameters)

