# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Language Class
@Author Yi Zhu
Upated 02.11.2017
"""

#************************************************************
# Imported Libraries
#************************************************************

class Lang:
  def __init__(self, name):
    self.name = name
    self.word2idx = {}
    self.word2count = {}
    self.idx2word = {0: 'SOS', 1: 'EOS'}
    self.n_words = 2 # Count SOS and EOS


  def addSentence(self, sentence):
    for word in sentence.strip().split(' '):
      self.addWord(word)


  def addWord(self, word):
    if word not in self.word2idx:
      self.word2idx[word] = self.n_words
      self.word2count[word] = 1
      self.idx2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1
