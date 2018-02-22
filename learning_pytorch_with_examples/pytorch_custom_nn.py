# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Pytorch example for customizing nn
@Author Yi Zhu
Upated 25/10/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
from torch.autograd import Variable


class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
      In the constructor we instantiate two nn.Linear modules and assign them as
      member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.Linear1 = torch.nn.Linear(D_in, H)
    self.Linear2 = torch.nn.Linear(H, D_out)


  def forward(self, x):
    """
      In the forward function we accept a Variable of input data and we must return
      a Variable of output data. We can use Modules defined in the constructor as
      well as arbitrary operators on Variables.
    """
    h_relu = self.Linear1(x).clamp(min = 0)
    y_pred = self.Linear2(h_relu)
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
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)

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
