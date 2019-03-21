import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) #(D,C)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape
  _, C = W.shape
  for i in range(num_train):
      scores = X[i].dot(W) #(C,)
      #prevent the instability of numpy
      scores -= np.max(scores)
      exp_scores = np.exp(scores) #(C,)

    
      score_true = exp_scores[y[i]]
      score_sum = np.sum(exp_scores)
      normalize_true = score_true/score_sum
      loss -= np.log(normalize_true )
      
      #gradient update
      exp_normalize = exp_scores/score_sum
      update = (X[i].reshape(dim,1)).dot(exp_normalize.reshape(1,C)) #(D,C)
      dW += update #X[i]: (D,)
 
      #for the true class
      dW[:,y[i]] -= X[i]
      
      
  
  #average the loss
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)
    
  #average dW
  dW /= num_train
  dW += reg*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) #(D, C)
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D = X.shape
  _, C = W.shape
  scores = X.dot(W) #(N,C)
  #prevent the instability of numpy
  scores -= np.max(scores, axis=1, keepdims=True)

  exp_scores = np.exp(scores) #(N,C)
  
  #find true score for each training point
  score_true = exp_scores[np.arange(N),y].reshape(N,1) #(N,1)
  score_sum = np.sum(exp_scores,axis = 1, keepdims=True) #(N,1)
  #element wise dividing
  normalize_true = score_true/score_sum
  loss -= np.sum(np.log(normalize_true ))
      
  #gradient update
  exp_normalize = exp_scores/score_sum #(N,C)
  #-1 for the value of true class
  exp_normalize[np.arange(N),y] -= 1
  dW = X.T.dot(exp_normalize) #(D,C)

     
  
  #average the loss
  loss /= N
  loss += 0.5 * reg * np.sum(W**2)
    
  #average dW
  dW /= N
  dW += reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

