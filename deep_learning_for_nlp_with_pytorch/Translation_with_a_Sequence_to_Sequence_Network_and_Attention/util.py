# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Util scripts
@Author Yi Zhu
Upated 06.11.2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import unicodedata
import re

from lang import Lang

import torch
from torch.autograd import Variable

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalizeString(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
  return s


def readLangs(lang1, lang2, reverse = False):
  print('Reading lines ...')
  # Read the file and split into lines
  f = open('data/{}-{}.txt'.format(lang1, lang2), 'r', encoding = 'utf-8')
  lines = f.read().strip().split('\n')

  # Split every line into pairs and normalize
  pairs = [[normalizeString(p) for p in l.strip().split('\t')] for l in lines]

  # Reverse pairs, make Lang instances
  if reverse:
    pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang(lang2)
    output_lang = Lang(lang1)
  else:
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

  return input_lang, output_lang, pairs

  
def filterPair(p):
  return (len(p[0].split(' ')) < MAX_LENGTH and 
          len(p[1].split(' ')) < MAX_LENGTH)


def filterPairs(pairs):
  return [p for p in pairs if filterPair(p)]
  

def prepareData(lang1, lang2, reverse = False):
  input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
  print('read {} sentence pairs'.format(len(pairs)))
  pairs = filterPairs(pairs)
  print('trimmed to {} sentence pairs'.format(len(pairs)))
  print('Counting words...')
  for p in pairs:
    input_lang.addSentence(p[0])
    output_lang.addSentence(p[1])
  print('Counted words:')
  print(input_lang.name, input_lang.n_words)
  print(output_lang.name, output_lang.n_words)
  return input_lang, output_lang, pairs

  
def indexesFromSentence(lang, sentence):
  return [lang.word2idx[word] for word in sentence.strip().split(' ')]


def variableFromSentence(lang, sentence, use_cuda):
  indexes = indexesFromSentence(lang, sentence)
  indexes.append(EOS_token)
  sent_var = Variable(torch.LongTensor(indexes))
  sent_var = sent_var.cuda() if use_cuda else sent_var
  return sent_var


def variablesFromPair(pair, input_lang, output_lang, use_cuda = False):
  input_variable = variableFromSentence(input_lang, pair[0], use_cuda)
  output_variable = variableFromSentence(output_lang, pair[1], use_cuda)
  return (input_variable, output_variable)
