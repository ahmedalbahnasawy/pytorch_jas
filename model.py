import numpy as np
import torch.nn.functional as F
import torch
import torch.nn
torch.manual_seed(1)


class RNNEncoder(torch.nn.Module):
  def __init__(self,n_layers,rnn_type='blstm',n_units=512,odims=1024, batch_first=True, num_dir =2, projection = True):
    super(RNNEncoder).__init__()


    def build_model(n_layers,rnn_type,n_units,odims, num_dir, projection,batch_first=True):

        rnn = []
        proj = []
        for i in range(n_layers):
            bidirectional = True if ('blstm' in rnn_type or 'lstm' in rnn_type) else False
            rnn_i = nn.LSTM
            rnn += [rnn_i(odims, n_units, 1, batch_first=True,
                                       bidirectional=bidirectional)]
            if projection:
                num_projection = 512
                if i != n_layers-1:
                    proj += [nn.Linear(n_units*num_dir, num_projection)]
        return rnn
    value = build_model(n_layers,rnn_type,n_units,odims, num_dir, projection,batch_first=True)
    print(value)

    "Not yet implemented "
class Decoder(torch.nn.Module):
  def __init__():
    super(RNNDecoder).__init__()
    "Not yet implemented "

class Build_loss(torch.nn.Module):
  def __init__(self, loss_tybe, loss_list):

    super(Build_loss).__init__()
        
    if loss_type not in loss_list:
      raise NotImplementedError
    else:
      if loss_type = 'transducer':
        from warprnnt_pytorch import RNNTLoss
        criterion = RNNTloss()
      else:
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)

class CNN_layerNorm(torch.nn.Module):
  "Layer Normalization after CNN layer"
  '''args: feats (batch, channel, feature, time'''
  def __init__(self, feats):
    super(CNN_layerNorm, self).__init__()
    self.layer_norm = torch.nn.LayerNorm(feats)
  def forward(self,x):
    x = x.transpose(x,2,3).contiguous()
    x = layer_norm(x)
    x = x.transpose(2, 3).contiguous()
    return x

class Resiudal_CNN(torch.nn.Module):
  "Resiudal CNN using skip connection technique"
  '''args: in_channel shape (batch,channel,feats,time)'''
  def __init__(self,feats):
    super(Resiudal_CNN,self).__init__()
    #self.input_channel = input_channel
    self.conv_1 = torch.nn.Conv2d(32, 64, kernel_size =3, stride =1 , padding=1 )
    self.drop_1 = torch.nn.Dropout2d(0.3)
    self.conv_2 = torch.nn.Conv2d(64, 64, kernel_size =3, stride =1 , padding=1 )
    self.drop_2 = torch.nn.Dropout2d(0.3)
    self.layer_norm_1 = CNN_layerNorm(feats)
    self.layer_norm_2 = CNN_layerNorm(feats)

  def forward(self, x):
    resiudal_conn = x
    x = self.layer_norm_1(x)
    x = F.relu(x)
    x = self.drop_1(x)
    x = self.conv_1(x)
    x = self.layer_norm_2(x)
    x = F.relu(x)
    x = self.drop_2(x)
    x = self.conv_2(x)
    x = x + resiudal_conn
    return x

   
   class BidirectionalGRU(torch.nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.bidirect = torch.nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=3, batch_first=batch_first, bidirectional=True)
        self.layer_norm = CNN_layerNorm(rnn_dim)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.bidirect(x)
        x = self.dropout(x)
        return x

l = BidirectionalGRU(512,512,0.2,batch_first=True)
inp = torch.randn(1024, 112, 8)

 class CNN_Network(torch.nn.Module):
  def __init__(self , input_channel):
    super(CNN_Network, self).__init__()
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
      x= batch_input.view(batch_input.size(0), batch_input.size(1), self.input,
                             batch_input.size(2) // self.input).transpose(1, 2)
      #x = batch_input.view()
      x = F.relu(self.bn_1(self.conv_1(batch_input)))
      x = F.relu(self.bn_2(self.conv_2(x)))
      x = F.max_pool2d(x,2,stride=2)
      x = F.relu(self.bn_3(self.conv_3(x)))
      x = F.relu(self.bn_4(self.conv_4(x)))
      x = F.max_pool2d(x,2,stride=2,ceil_mode=True)
      # input length should always be numpy array
      input_length = np.array(input_length,dtype=np.float32)
      input_length = np.array(np.ceil(input_length/2), dtype=np.int64)
      
      
      x = x.transpose(1, 2)
      return x ,  input_length
  

  
 class ASR(torch.nn.Module):
  def __init__(self , input_channel):
    super(ASR, self).__init__()
    self.Encoder = RNNEncoder(rnn_tybe,input_dim,num_units, num_layer,dropout_in)
    self.Decoder = RNNDecoder()
    self.loss = Build_loss(args.model)
  
  
f = torch.ones(1,3,40,3)
m = Resiudal_CNN(f)


input_b = torch.randn(83)
model = CNN_Network(input_b)
