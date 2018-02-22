# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Classifying Names with a Character-Level RNN
@Author Yi Zhu
Upated 31/10/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)

import glob
import unicodedata
import string
import random




#Preparing the Data
def findFiles(path):
  return glob.glob(path)

print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)
char_to_idx = {}
for c in all_letters:
  if c not in char_to_idx:
    char_to_idx[c] = len(char_to_idx)
  

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn' and c in all_letters)

#print(unicodeToAscii('Ślusàrski'))  

# Build the category_lines dictionary, a list of names per language
category_lines = {}
cat_to_idx = {}

# Read a file and split into lines
def readLines(filename):
  lines = open(filename, 'r').read().strip().split('\n')
  return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
  category = filename.split('/')[-1].split('.')[0]
  cat_to_idx[category] = len(cat_to_idx)
  lines = readLines(filename)
  category_lines[category] = lines

idx_to_cat = {v: k for k, v in cat_to_idx.items()}
n_sample = sum([len(v) for k, v in category_lines.items()])
#print(n_sample)
for k, v in category_lines.items():
  random.shuffle(v)
train_por = 0.95 # meaning 8 / 10 of the data will be seen as training data
category_lines_train = {k: v[: int(len(v) * train_por)] for k, v in category_lines.items()}
category_lines_test = {k: v[int(len(v) * train_por):] for k, v in category_lines.items()}
n_train = sum([len(v) for k, v in category_lines_train.items()])
n_test =  sum([len(v) for k, v in category_lines_test.items()])



# Turning Names into Variables
def prepareStr(s, char_to_idx):
  idxs = [char_to_idx[c] for c in s]
  idx_t = torch.LongTensor(idxs)
  return autograd.Variable(idx_t)

#print(prepareStr('Jones', char_to_idx))

def prepareCat(category, cat_to_idx):
  return autograd.Variable(torch.LongTensor([cat_to_idx[category]]))

#print(prepareCat('Chinese', cat_to_idx))


# Creating the Network
class RNNModel(nn.Module):
  def __init__(self, vocab_size, char_dim, hidden_dim, target_size):
    super(RNNModel, self).__init__()

    self.lstm_num_layers = 4
    self.char_embedding = nn.Embedding(vocab_size, char_dim)
    self.lstm = nn.LSTM(char_dim, hidden_dim, num_layers = self.lstm_num_layers, bidirectional = True)
    self.h2o = nn.Linear(2 * hidden_dim, target_size)

    self.hidden_dim = hidden_dim
    self.hidden = self.init_hidden()

  
  def init_hidden(self):
    return (autograd.Variable(torch.zeros(self.lstm_num_layers * 2, 1, self.hidden_dim)),
            autograd.Variable(torch.zeros(self.lstm_num_layers * 2, 1, self.hidden_dim)))
  

  def forward(self, inputs):
    self.hidden = self.init_hidden()
    embeds = self.char_embedding(inputs).view(len(inputs), 1, -1)
    lstm_out, self.hidden = self.lstm(embeds, self.hidden)
    #print(self.hidden[0].size())
    #print(lstm_out.size())
    out = self.h2o(lstm_out[-1].squeeze())
    log_probs = F.log_softmax(out)
    return log_probs 


CHAR_DIM = 10
HIDDEN_DIM = 128
n_cat = len(cat_to_idx)
n_char = len(char_to_idx)

# Run before training
model = RNNModel(n_char, CHAR_DIM, HIDDEN_DIM, n_cat)
inputs = prepareStr('Albert', char_to_idx)

log_probs = model(inputs)
print(idx_to_cat[log_probs.max(0)[1].data[0]])


# Get random training example
def randomChoice(l):
  return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(cat_lines):
  category = randomChoice(list(cat_to_idx.keys()))
  category_v = autograd.Variable(torch.LongTensor([cat_to_idx[category]]))
  line = randomChoice(cat_lines[category])
  line_v = prepareStr(line, char_to_idx)
  return category, line, category_v, line_v


loss_function = nn.NLLLoss()
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

print_every = 100

for epoch in range(10):
  ct = 0 
  train_acc = .0
  while ct < n_train:
    model.zero_grad()

    category, line, category_v, line_v = randomTrainingExample(category_lines_train)

    log_probs = model(line_v).view(1, -1)
    loss = loss_function(log_probs, category_v)
    loss.backward()

    optimizer.step()
    ct += 1
    train_acc += 1 if log_probs.max(1)[1].data[0] == cat_to_idx[category] else 0

    if ct % print_every == 0:
      print('Epoch {}, {}/{}'.format(epoch + 1, ct, n_train))
      print('Training Accuracy: {:.3f}%'.format(train_acc * 100 / print_every))
      train_acc = 0

      test_acc = 0
      for cat, names in category_lines_test.items():
        cat_idx = cat_to_idx[cat]
        for name in names:
          name_v = prepareStr(name, char_to_idx)
          log_probs = model(name_v).view(1, -1)
          test_acc += 1 if log_probs.max(1)[1].data[0] == cat_idx else 0
      print('Test Accuracy: {:.3f}%'.format(test_acc * 100 / n_test))




  
