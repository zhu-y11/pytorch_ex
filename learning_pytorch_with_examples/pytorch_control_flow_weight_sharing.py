# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Pytorch example for control flow and weight sharing
@Author Yi Zhu
Upated 25/10/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import random
import torch
from torch.autograd import Variable


class DynamicNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
      In the constructor we instantiate two nn.Linear modules and assign them as
      member variables.
    """
    super(DynamicNet, self).__init__()
    self.input_layer = torch.nn.Linear(D_in, H)
    self.middle_layer = torch.nn.Linear(H, H)
    self.output_layer = torch.nn.Linear(H, D_out)


  def forward(self, x):
    """
      For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
      and reuse the middle_linear Module that many times to compute hidden layer
      representations.

      Since each forward pass builds a dynamic computation graph, we can use normal
      Python control-flow operators like loops or conditional statements when
      defining the forward pass of the model.

      Here we also see that it is perfectly safe to reuse the same Module many
      times when defining a computational graph. This is a big improvement from Lua
      Torch, where each Module could be used only once.
    """
    h_relu = self.input_layer(x).clamp(min = 0)
    for _ in range(random.randint(0, 3)):
      h_relu = self.middle_layer(h_relu).clamp(min = 0)
    y_pred = self.output_layer(h_relu)
    return y_pred




dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in).type(dtype))
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad = False)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)

for t in range(500):
  # Forward pass: Compute predicted y by passing x to the model 
  y_pred = model(x)

  # Compute and print loss. We pass Variables containing the predicted and true
  # values of y, and the loss function returns a Variable containing the
  # loss.
  loss = criterion(y_pred, y)
  print(t, loss.data[0])
  
  # Zero gradients, perform a backward pass, and update the weights. 
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
