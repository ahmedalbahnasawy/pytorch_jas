
@Author: Ahmed Albhnasawy
@Email: ahmedbhna@gmail.com

r'''optimizer'''
import torch
import torch.nn as nn
from torch.nn import functional as F 

    
class ASR_optimizer(object):
    def __init__(self,model, params):
        super(ASR_optimizer, self).__init__()
        self.params = params
        self.model = model
        self.parallel_mode = self.params['parallel_mode']
        self.lr = self.params['lr']

        if params['optimizer'] == 'sgd':
          self.optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr = self.lr, momentum=0.9)
        elif params['optimizer'] == 'adam':
          self.optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),lr = self.lr, betas=(0.9,0.98), eps=1e-8)
        elif params['optimizer'] == 'adadelta':
          self.optimizer = torch.optim.Adadelta(filter(lambda p:p.requires_grad, model.parameters()),lr = self.lr, eps=1e-8)
        else:
          raise NotImplementedError
        
  def state_dict(self):
      return self.optimizer.state_dict()

  def load_state_dict(Self,state_dict):
      self.optimizer.load_state_dict(state_dict)

  def zero_grad(self):
      self.optimizer.zero_grad()
