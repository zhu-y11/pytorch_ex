# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Main script of word language model, reimplemented according to pytorch example
@Author Yi Zhu
Upated 20/10/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math

import data
import model

#************************************************************
# Main Function
#************************************************************
def main(args):
  # Set the random seed manually for reproducibility.
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    if not args.cuda:
      print("WARNING: You have a CUDA device, so you should probably run with --cuda") 
    else:
      torch.cuda.manual_seed(args.seed)
  
  # load the data
  """
    I like it <eos> It was cool <eos> do you know <eos>
    batch_size = 3, nbatch = 4
  """
  corpus = data.Corpus(args.data)
  eval_batch_size = 10
  # nbatch, batch_size
  train_data = batchify(corpus.train, args.batch_size)
  test_data = batchify(corpus.test, eval_batch_size)
  val_data = batchify(corpus.valid, eval_batch_size)
  """
    I     It    do
    like  was   you
    it    cool  knoe
    <eos> <eos> <eos>
  """

  print(corpus.train.size())
  print(corpus.valid)
  print(val_data)
  print('Training data: {}'.format(train_data.size()))
  print('Test data: {}'.format(test_data.size()))
  print('Valid data: {}'.format(val_data.size()))

  # build the model
  ntokens = len(corpus.dictionary)
  nn_model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
  if args.cuda:
    nn_model.cuda()

  criterion = nn.CrossEntropyLoss()

  # Training
  lr = args.lr
  best_val_loss = None

  try:
    # Loop over epochs.
    for epoch in range(1, args.epochs + 1):
      epoch_start_time = time.time()
      train(ntokens, nn_model, args, criterion, epoch, lr, train_data)

      val_loss = evaluate(val_data, nn_model, ntokens, eval_batch_size, args, criterion)
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
            val_loss, math.exp(val_loss)))
      print('-' * 89)
      if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
          torch.save(nn_model, f)
        best_val_loss = val_loss
      else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0
  except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

  # Load the best saved model.
  with open(args.save, 'rb') as f:
      nn_model = torch.load(f)

  # Run on test data.
  test_loss = evaluate(test_data, nn_model, ntokens, eval_batch_size, args, criterion)
  print('=' * 89)
  print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
  print('=' * 89) 


#************************************************************
# Batchify input corpus
#************************************************************
def batchify(data, batch_size):
  # Get batch number for the data
  nbatch = data.size(0) // batch_size
  # Trim off any extra elements that wouldn't cleanly fit (remainders).
  data = data.narrow(0, 0, nbatch * batch_size)
  # Evenly divide the data across the bsz batches.
  # dim: (nbatch, batch_size)
  data = data.view(batch_size, -1).t().contiguous()
  if args.cuda:
    data = data.cuda()
  return data 
  

#************************************************************
# Training
#************************************************************
def train(ntokens, nn_model, args, criterion, epoch, lr, train_data):
  # Turn on training mode which enables dropout.
  nn_model.train()
  total_loss = 0
  start_time = time.time()
  hidden = nn_model.init_hidden(args.batch_size)

  # data: (nbatch, batch_size)
  for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    """
      assume bptt = 2
      i = 0, 2
    """
    # seq_len, batch_size
    input_data, targets = get_batch(train_data, i, args.bptt)
    """
      i = 1     
      I     It    do
      like  was   you

      i = 2
      it    cool  know
      <eos> <eos> <eos>
    """
    # Starting each batch, we detach the hidden state from how it was previously produced.
    # If we didn't, the model would try backpropagating all the way to start of the dataset.
    hidden = repackage_hidden(hidden)
    nn_model.zero_grad()
    # call forward function
    output, hidden = nn_model(input_data, hidden)
    loss = criterion(output.view(-1, ntokens), targets)
    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm(nn_model.parameters(), args.clip)
    for p in nn_model.parameters():
      p.data.add_(-lr, p.grad.data)

    total_loss += loss.data

    if batch % args.log_interval == 0 and batch > 0:
      cur_loss = total_loss[0] / args.log_interval
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, batch, len(train_data) // args.bptt, lr,
            elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()
  

def get_batch(source, i, bptt, evaluation = False):
  seq_len = min(bptt, len(source) - 1 - i)
  # seq_len, batch_size
  data = Variable(source[i: i + seq_len], volatile = evaluation)
  target = Variable(source[i + 1: i + 1 + seq_len].view(-1))
  return data, target


def repackage_hidden(h):
  """
    Wraps hidden states in new Variables, to detach them from their history.
  """
  if type(h) == Variable:
      return Variable(h.data)
  else:
      return tuple(repackage_hidden(v) for v in h)


#************************************************************
# Evaluate
#************************************************************
def evaluate(data_source, nn_model, ntokens, eval_batch_size, args, criterion):
  # Turn on evaluation mode which disables dropout.
  nn_model.eval()
  total_loss = 0
  hidden = nn_model.init_hidden(eval_batch_size)
  for i in range(0, data_source.size(0) - 1, args.bptt):
    input_data, targets = get_batch(data_source, i, args.bptt, evaluation = True)
    output, hidden = nn_model(input_data, hidden)
    output_flat = output.view(-1, ntokens)
    total_loss += input_data.size(0) * criterion(output_flat, targets).data
    hidden = repackage_hidden(hidden)
  return total_loss[0] / data_source.size(0)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Reimplementation of PyTorch PennTreeBank RNN/LSTM Language Model')
  parser.add_argument('--data', type = str, default = './data/penn',
                      help = 'data repo')
  parser.add_argument('--seed', type = int, default = 1234,
                      help = 'random seed')
  parser.add_argument('--cuda', action = 'store_true',
                      help = 'use CUDA')
  parser.add_argument('--batch_size', type = int, default = 20, metavar = 'N',
                      help = 'use CUDA')
  parser.add_argument('--model', type = str, default = 'LSTM',
                      help = 'type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
  parser.add_argument('--emsize', type = int, default = 200,
                      help = 'size of word embeddings')
  parser.add_argument('--nhid', type = int, default = 200,
                      help = 'number of hidden units per layer')
  parser.add_argument('--nlayers', type = int, default = 2,
                      help = 'number of layers')
  parser.add_argument('--dropout', type = float, default = 0.2,
                      help = 'droput applied to layers (0 = no dropout)')
  parser.add_argument('--tied', action = 'store_true',
                      help = 'tie the word embedding and softmax weights')
  parser.add_argument('--lr', type = float, default = 20,
                      help = 'initial learning rate')
  parser.add_argument('--epochs', type = int, default = 40,
                      help = 'upper epoch limit')
  parser.add_argument('--bptt', type = int, default = 35,
                    help = 'sequence length')
  parser.add_argument('--log-interval', type = int, default = 100, metavar = 'N',
                    help = 'report interval')
  parser.add_argument('--clip', type = float, default = 0.25,
                      help = 'gradient clipping')
  parser.add_argument('--save', type = str,  default = 'model.pt',
                      help = 'path to save the final model')
  args = parser.parse_args()
  main(args)  
