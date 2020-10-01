import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAtt(nn.Module):
    """ Multi Head Attention layer """
    def __init__(self, n_head,n_embd, dropout):
        """ n_head : number of head 
            n_embd: number of features
            dropout: dropout prob
        """

        super(MultiHeadAtt,self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.d_k = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # output projection
        self.dropout_output = nn.Dropout(p=dropout)
        self.linear_output = nn.Linear(n_embd, n_embd)
        self.attn = None
        
    def forward(self, query, key, value):
        """Compute vector using Dot product attention
           Args: query, key and value (torch.Tensor) shape( #batch, time, size)
           returns Transformed query, key and value tensor of shape (#batch, n_head,time , n_embd // n_head) """
        batch = query.size(0)
        q = self.query(query).view(batch, -1, n_head, self.d_k)
        k = self.key(key).view(batch, -1 , n_head , self.d_k)
        v = self.value(value).view(batch, -1, n_head , self.d_k)
        q = q.transpose(1,2)  
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # we are using self Att without mask
        self.attn = torch.softmax(score, dim =1)
        out = self.dropout_output(self.attn)
        x = torch.matmul(out, value)  
        x = (
            x.transpose(1, 2).contiguous().view(batch, -1, n_head * self.d_k)
        )

        return self.linear_out(x)  # (batch, time1, d_model)

