import torch
class ASR_optim(optim):
    def__init__(self,model,params, parallel_mode = 'dp'):
        super(ASR_optim).__init__()
        self.params = params
        self.model = model
        self.parallel_mode = parallel_mode
        self.lr = self.parms['lr']
        if params['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr = self.lr, betas=(0.9,0.98), eps=1e-8)
  
