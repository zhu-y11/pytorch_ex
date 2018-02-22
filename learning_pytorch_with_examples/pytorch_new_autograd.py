# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Pytorch example for new autograd function
@Author Yi Zhu
Upated 25/10/2017
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
from torch.autograd import Variable

class MyReLu(torch.autograd.Function):
  """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
  """

  def forward(self, input_data):
    """
      In the forward pass we receive a Tensor containing the input and return a
      Tensor containing the output. You can cache arbitrary Tensors for use in the
      backward pass using the save_for_backward method.
    """
    self.save_for_backward(input_data)
    return input_data.clamp(min = 0)


  def backward(self, grad_output):
    """
      In the backward pass we receive a Tensor containing the gradient of the loss
      with respect to the output, and we need to compute the gradient of the loss
      with respect to the input.
    """
    input_data,  = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input_data < 0] = 0
    return grad_input



dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad = False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad = False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad = True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad = True)

learning_rate = 1e-6

for t in range(500):
  # Construct an instance of our MyReLU class to use in our network
  relu = MyReLu()
  
  # Forward pass: compute predicted y using operations on Variables; we compute
  # ReLU using our custom autograd operation.
  y_pred = relu(x.mm(w1)).mm(w2)

  # Compute and print loss using operations on Variables.
  # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
  # (1,); loss.data[0] is a scalar value holding the loss.
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.data[0])

  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Variables with requires_grad=True.
  # After this call w1.grad and w2.grad will be Variables holding the gradient
  # of the loss with respect to w1 and w2 respectively.
  loss.backward()

  # Update weights using gradient descent; w1.data and w2.data are Tensors,
  # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
  # Tensors.
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data
  
  # Manually zero the gradients after updating weights
  w1.grad.data.zero_()
  w2.grad.data.zero_()
