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
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    score_i = X[i].dot(W)
    score_max = np.max(score_i)
    loss += score_max - score_i[y[i]] + np.log(np.sum(np.exp(score_i - score_max)))
    dW_i = np.outer(X[i], np.exp(score_i - score_max)/(np.sum(np.exp(score_i - score_max))))
    dW_i[:, y[i]] = 0
    dW += dW_i/num_train
    dW[:, y[i]] += (np.exp(score_i[y[i]] - score_max)/(np.sum(np.exp(score_i - score_max))) - 1) * X[i]/num_train
  loss = loss/num_train + 0.5 * reg * np.sum(W * W)
  dW += reg * W
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
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  score_matrix = X.dot(W)
  score_max = np.max(score_matrix, axis=1)
  score_matrix_reducted = score_matrix - score_max.reshape(-1, 1)
  score_non_log = np.exp(score_matrix_reducted)
  score_unify = score_non_log/np.sum(score_non_log, axis=1).reshape(-1,  1)
  score_unify[np.arange(num_train), y] -= 1
  dW = X.T.dot(score_unify)/num_train + reg * W
  loss_col = score_max - score_matrix[np.arange(num_train), y] + np.log(np.sum(score_non_log, axis=1))
  loss = np.sum(loss_col)/num_train + 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

