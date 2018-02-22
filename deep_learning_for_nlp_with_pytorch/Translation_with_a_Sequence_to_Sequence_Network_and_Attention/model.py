# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Seq2Seq Model
@Author Yi Zhu
Upated 06.11.2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class EncoderRNN(nn.Module):
  def __init__(self, input_vocab_size, input_embedding_dim, hidden_dim, n_layers = 2, use_cuda = False):
    super(EncoderRNN, self).__init__()
    self.use_cuda = use_cuda
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.embedding_dim = input_embedding_dim

    self.embedding = nn.Embedding(input_vocab_size, input_embedding_dim)
    self.lstm = nn.LSTM(input_embedding_dim, hidden_dim, n_layers, bidirectional = False)
    self.hidden = self.init_hidden()


  def init_hidden(self):
    results = (Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)),
               Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)))
    if self.use_cuda:
      results = results.cuda()
    return results
 

  def forward(self, inputs):
    print(inputs.size())
    embeds = self.embedding(inputs).view(inputs.size()[0], 1, self.embedding_dim)
    print(embeds.size())
    lstm_out, lstm_hidden = self.lstm(embeds, self.hidden)
    print(lstm_out.size(), lstm_hidden[0].size(), lstm_hidden[1].size())
    return lstm_out, lstm_hidden_cell



class DecoderRNN(nn.Module):
  def __init__(self, output_vocab_size, output_embedding_dim, hidden_dim, n_layers = 2, use_cuda = False):
    super(DecoderRNN, self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding_dim = output_embedding_dim
    self.n_layers = n_layers
    self.use_cuda = use_cuda

    self.embedding = nn.Embedding(output_vocab_size, output_embedding_dim)
    self.lstm_cell = nn.LSTMCell(output_embedding_dim, hidden_dim)
    self.linear = nn.Linear(hidden_dim, output_vocab_size)
    self.hidden = self.init_hidden()


  def init_hidden(self):
    results = (Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)),
               Variable(torch.zeros(self.n_layers, 1, self.hidden_dim)))
    if self.use_cuda:
      results = results.cuda()
    return results


  def forward(self, inputs, hidden_cell):
    embed = self.embedding(inputs).view(1, 1, -1)
    for i in range(self.n_layers):
      hidden_cell = self.lstm_cell(embed, hidden_cell)
    out = self.linear(hidden_cell[0])
    out = F.log_softmax(out) 
    return out, hidden_cell
