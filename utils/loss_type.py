import torch
import torch.nn as nn
torch.manual_seed(1)

class Build_loss():
  def __init__(self,loss_type):
    self.loss_type = loss_type

  def which_loss(self, loss_type):
    self.loss_type = loss_type
    if loss_type not in ['ctc', 'transducer']:
      raise NotImplementedError
    else:
      if loss_type =='transducer':
        from warprnnt_pytorch import RNNTLoss
        criterion = RNNTloss()
        criterion = torch.nn.CrossEntropyLoss()
      else:
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
      return criterion

  def test_dummy(self, critertion):
    self.criterion = criterion
    input_seq_len = 3 #input seq length
    num_classes = 2 # vocab including <CTC blank>
    bs = 1 #batch_size
    s = 30 #target_Seq_length
    dummy_input = torch.randn(input_seq_len, bs, num_classes).log_softmax(2).detach().requires_grad_()
    target = torch.randint(low=1, high=num_classes, size=(bs, s), dtype=torch.long)
    input_lengths = torch.full(size=(bs,), fill_value=1, dtype=torch.long)
    target_lengths = torch.randint(low=20, high=s, size=(bs,), dtype=torch.long)
    loss = criterion(dummy_input, target, input_lengths, target_lengths)

    return dummy_input


loss = Build_loss(loss_type='ctc')
loss = loss.which_loss(loss_type='ctc')
