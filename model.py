import numpy as np
import torch.nn.functional as F
import torch
import torch.nn
class STT(torch.nn.Module):
  def __init__(self , input_channel):
    super(STT, self).__init__()
    self.input_channel = input_channel
    self. conv_1 = torch.nn.Conv2d(input, 64, kernel_size =3, stride =1 , padding=1 )
    self.bn_1 = torch.nn.BatchNorm2d(64 , eps=1e-8, momentum=0.1, affine=True, Track_running_stats=True)
    self.conv_2 = torch.nn.Conv2d(64,64,kernel_size =3, stride =1, padding =1)
    self.bn_2 = torch.nn.BatchNorm2d(64, eps=1e-8, momentum=0.1, affline=True)
    self.conv_3 = torch.nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
    self.bn_3 = torch.nn.BatchNorm2d(128, eps=1e-8, momentum=0.1,affine=True, track_running_stats=True)
    self.conv_4 = torch.nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1)
    self.bn_4 = torch.nn.BatchNorm2d(128,eps=1e-8, momentum=0.1, affine=True, track_running_stats=True)



    def forward(self, batch_input, input_length, **kwargs):
      # input mini batch of data
      #x = batch_input.view()
      x = F.relu(self.bn_1(self.conv_1(batch_input)))
      x = F.relu(self.bn_2(self.conv_2(x)))
      x = F.max_pool2d(x,2,stride=2)
      x = F.relu(self.bn_3(self.conv_3(x)))
      x = F.relu(self.bn_4(self.conv_4(x)))
      x = F.max_pool2d(x,2,stride=2,ceil_mode=True)

      return 
input_b = torch.randn(83)
model = STT(input_b)

class Encoder(torch.nn.Module):
  def __init__ ():
    super(Encoder).__()
    "Not yet implemented "
class Decoder(torch.nn.Module):
  def __init__():
    super(Decoder).__init__()
    "Not yet implemented "

class Transducer(torch.nn.Module):
  def __init__():
    super(Transducer).__init__()
    "Not yet implemented "

class CTC(torch.nn.Module):
  def __init__():
    super(CTC).__init__()
    "Not yet implemented "

