# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Translation with a Sequence to Sequence Network and Attention
@Author Yi Zhu
Upated 06.11.2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import util
import model

use_cuda = torch.cuda.is_available()


input_lang, output_lang, pairs = util.prepareData('eng', 'fra', True)
#print(random.choice(pairs))


teacher_forcing_ratio = 0.5
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
    criterion, max_length = util.MAX_LENGTH, ):

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = input_variable.size()[0]
  target_length = target_variable.size()[0]

  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_dim))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
  
  loss = 0
  
  encoder_out, encoder_hidden = encoder(input_variable)
  for ei in range(max_length):
    encoder_outputs[ei] = encoder_out[ei][0]

  decoder_input = Variable(torch.LongTensor([[util.SOS_token]]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input

  decoder_hidden = encoder_hidden

  #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
  use_teacher_forcing = True

  if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
      loss +=  criterion(decoder_output, target_variable[di])
      decoder_input = target_variable[di] # Teacher forcing
  else:
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
      loss += criterion(decoder_output, target_variable[di])    
      topv, topk = decoder_output.data.topk(1)
      ni = topk[0][0]
      
      decoder_input = Variable(torch.LongTensor([[ni]]))
      decoder_input = decoder_input.cuda() if use_cuda else decoder_input

      if ni == util.EOS_token:
        break

  loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0] / target_length


def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every = 100, learning_rate = 1e-2):
  start = time.time()
  print_loss_total = 0 # Reset every print_every

  encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
  training_pairs = [util.variablesFromPair(random.choice(pairs), input_lang, output_lang, use_cuda) for _ in range(n_iters)]
  criterion = nn.NLLLoss()

  for n_iter in range(1, n_iters + 1):
    training_pair = training_pairs[n_iter - 1]
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    loss = train(input_variable, target_variable, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss

    if n_iter % print_every:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%s (%d %d%%) %.4f' % (timeSince(start, n_iter / n_iters),
            n_iter, n_iter / n_iters * 100, print_loss_avg))
      evaluateRandomly(encoder1, attn_decoder1)
      


def evaluate(encoder, decoder, sentence, max_length = util.MAX_LENGTH):
  input_variable = util.variableFromSentence(input_lang, sentence, use_cuda)
  input_length = input_variable.size()[0]      

  encoder.hidden = encoder.init_hidden()

  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_dim))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

  encoder_out, encoder_hidden = encoder(input_variable)
  for ei in range(max_length):
    encoder_outputs[ei] = encoder_out[ei][0]  

  decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input
  decoder_hidden = encoder_hidden

  decoder_words = []

  for di in range(max_length):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]

    if ni == util.EOS_token:
      decoded_words.append('<EOS>')
      break
    else:
      decoded_words.append(output_lang.idx2word(ni))

    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

  return decoded_words


def evaluateRandomly(encoder, decoder, n = 10):
  for i in range(n):
    pair = random.choice(pairs)
    print('>', pair[0])
    print('=', pair[1])
    output_words = evaluate(encoder, decoder, pair[0])
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')




hidden_dim = 256
embedding_dim = 100

encoder_1 = model.EncoderRNN(input_lang.n_words, embedding_dim, hidden_dim)
decoder_1 = model.DecoderRNN(output_lang.n_words, embedding_dim, hidden_dim)
#attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, 1, dropout_p=0.1)

if use_cuda:
    encoder_1 = encoder_1.cuda()
    decoder_1 = decoder_1.cuda()
    #attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder_1, decoder_1, 75000)
