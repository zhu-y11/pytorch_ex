# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Sequence Models and Long-Short Term Memory Networks
@Author Yi Zhu
Upated 29/10/2017
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


lstm = nn.LSTM(3, 3) # Input dim is 3, output dim is 3
inputs = [autograd.Variable(torch.randn(1, 3)) for _ in range(5)] # make a sequence of length 5

# initialize the hidden state.
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          autograd.Variable(torch.randn(1, 1, 3)))

for i in inputs:
  # Step through the sequence one element at a time.
  # after each step, hidden contains the hidden state.
  out, hidden = lstm(i.view(1, 1, -1), hidden)


# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn(1, 1, 3))) # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out.size())
print(hidden)




#Example: An LSTM for Part-of-Speech Tagging

# Prepare data
def prepare_sequence(seq, to_ix):
  idxs = [to_ix[w] for w in seq]
  tensor = torch.LongTensor(idxs)
  return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

word_to_ix = {}
for sent, tags in training_data:
  for word in sent:
    if word not in word_to_ix:
      word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
    super(LSTMTagger, self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)

    # The LSTM takes word embeddings as inputs, and outputs hidden states
    # with dimensionality hidden_dim.
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = True)

    # The linear layer that maps from hidden state space to tag space
    self.hidden2tag = nn.Linear(2 * hidden_dim, target_size)
    self.hidden  = self.init_hidden()


  def init_hidden(self):
    # Before we've done anything, we dont have any hidden state.
    # Refer to the Pytorch documentation to see exactly
    # why they have this dimensionality.
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),
            autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))


  def forward(self, sentence):
    embeds = self.word_embeddings(sentence).view(len(sentence), 1, -1)
    lstm_out, self.hidden = self.lstm(embeds, self.hidden) 
    print(lstm_out.size())
    print(self.hidden[0].size())
    tag_space = self.hidden2tag(lstm_out)
    tag_scores = F.log_softmax(tag_space).squeeze()
    return tag_scores


# Training
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(10): # again, normally you would NOT do 300 epochs, it is toy data
  total_loss = torch.Tensor([0])
  for sentence, tags in training_data:
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    optimizer.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden()

    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Variables of word indices.
    sentence_in = prepare_sequence(sentence, word_to_ix)
    targets = prepare_sequence(tags, tag_to_ix)

    # Step 3. Run our forward pass.
    tag_scores = model(sentence_in)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(tag_scores, targets)
    loss.backward()
    optimizer.step()
    total_loss += loss.data
  print(total_loss)

# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)

print('Sentence: {}\n\
       Predicted Tag: {}\n\
       Real Tag: {}'.format(training_data[0][0], ' '.join([ix_to_tag[i] for i in tag_scores.max(1)[1].data]), ' '.join(training_data[0][1])))




#Exercise: Augmenting the LSTM part-of-speech tagger with character-level features
print('*' * 50)
print('Exercise: Augmenting the LSTM part-of-speech tagger with character-level features')
print('*' * 50)
