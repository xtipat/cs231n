from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim) #(D,H)
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim,num_classes) #(H,C)
        self.params['b2'] = np.zeros((num_classes,))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #keep the value in local variable
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        #first layer and relu
        out1, cache1 = affine_forward(X,W1, b1)
        out1_relu, cache1_relu = relu_forward(out1)
        
        #second layer
        out2, cache2 = affine_forward(out1_relu, W2, b2)
        
        #score is not include softmax
        scores = out2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #compute the softmax loss
        loss, dscores = softmax_loss(scores, y)
        #add L2 regularization
        loss += 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2))
        
        #backpropagation
        dx2, dW2, db2 = affine_backward(dscores, cache2)
        grads['W2'] = dW2
        grads['b2'] = db2
        dx_relu = relu_backward(dx2, cache1_relu) #relu layer
        dx1, dW1, db1 = affine_backward(dx_relu, cache1)
        grads['W1'] = dW1
        grads['b1'] = db1
        
        #add L2 regualar
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        #keep the hidden layer size as well
        self.hidden_dims = hidden_dims

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        #For the hidden layer
        for i in range(len(hidden_dims)):
            #for the first layer
            if i == 0:
                
                self.params['W'+str(i)] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
                self.params['b'+str(i)] = np.zeros((hidden_dims[i],)) 
                #initialize for batch normalization
                if self.normalization=='batchnorm':
                    self.params['gamma'+str(i)] = np.ones((hidden_dims[i],))
                    self.params['beta'+str(i)] = np.zeros((hidden_dims[i],))
            else:
                self.params['W'+str(i)] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
                self.params['b'+str(i)] = np.zeros((hidden_dims[i],)) 
                #initialize for batch normalization
                if self.normalization=='batchnorm':
                    self.params['gamma'+str(i)] = np.ones((hidden_dims[i],))
                    self.params['beta'+str(i)] = np.zeros((hidden_dims[i],))
        #for the last layer with output
        self.params['W_last'] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params['b_last'] = np.zeros((num_classes,)) 
        #for key, item in self.params.items():
        #    print("key: {}, shape:{}".format(key, item.shape))

       
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #make the dictionary to store the name
        outs = {}
        caches = {}
        outs_relu = {}
        caches_relu= {}
        outs_batch = {}
        caches_batch = {}
        outs_dropout = {}
        caches_dropout = {}
        
        #The layer will have affine follows by relu
        for i in range(len(self.hidden_dims)):
            
            #the first layer has the the input X
            if i == 0:
                outs[str(i)], caches[str(i)] = affine_forward(X,
                                                              self.params['W'+str(i)],
                                                              self.params['b'+str(i)])
 
            #other than the first layer
            else: 
                outs[str(i)], caches[str(i)] = affine_forward(outs_relu[str(i-1)],
                                                              self.params['W'+str(i)],
                                                              self.params['b'+str(i)])
            #----    
            #batch normalization
            if self.normalization=='batchnorm':
                outs_batch[str(i)],caches_batch[str(i)]  = batchnorm_forward(outs[str(i)],
                                                                             self.params['gamma'+str(i)], 
                                                                             self.params['beta'+str(i)], 
                                                                             self.bn_params[i])                                 
                #relu
                outs_relu[str(i)], caches_relu[str(i)] = relu_forward(outs_batch[str(i)])
            #no batch norm just apply relu
            else:
                outs_relu[str(i)], caches_relu[str(i)] = relu_forward(outs[str(i)])
                
            #---
            #dropout (drop out follow the relu
            if self.use_dropout:
                outs_dropout[str(i)], caches_dropout[str(i)] = dropout_forward(outs_relu[str(i)],
                                                                              self.dropout_param)
                
                                                                              
        #the last layer with dropout      
        if self.use_dropout:  
            outs['last'], caches['last'] = affine_forward(outs_dropout[str(len(self.hidden_dims)-1)],
                                                          self.params['W_last'],
                                                          self.params['b_last'])            
        #the last layer without dropout
        else:
            outs['last'], caches['last'] = affine_forward(outs_relu[str(len(self.hidden_dims)-1)],
                                                          self.params['W_last'],
                                                          self.params['b_last'])
        scores = outs['last']
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #compute the softmax loss
        loss, dscores = softmax_loss(scores, y)
        #add L2 regularization
        for i in range(len(self.hidden_dims)):
            loss += 0.5*self.reg*(np.sum(self.params['W'+str(i)]**2))
        loss += 0.5*self.reg*(np.sum(self.params['W_last']**2))
        
        #for keep the value
        dx = {}
        dx_relu = {}
        dx_batch = {}
        dx_dropout = {}
        
        #backpropagation
        #the last layer (doesnot have relu function and normalization)
        dx['last'], grads['W_last'], grads['b_last'] = affine_backward(dscores, caches['last'])
         
    
        for i in reversed(range(len(self.hidden_dims))):
            
            #Dropout layer
            if self.use_dropout:
                if i == (len(self.hidden_dims) -1):
                    dx['last'] = dropout_backward(dx['last'], caches_dropout[str(i)]) 
                else:
                    dx['last'] = dropout_backward(dx[str(i+1)], caches_dropout[str(i)]) 

            if i == (len(self.hidden_dims) -1):
                dx_relu[str(i)] =  relu_backward(dx['last'], caches_relu[str(i)])
            else:
                dx_relu[str(i)] =  relu_backward(dx[str(i+1)], caches_relu[str(i)])
            
            #batch normalization layer
            if self.normalization=='batchnorm':
                dx_batch[str(i)], grads['gamma'+str(i)], grads['beta'+str(i)]= batchnorm_backward_alt(dx_relu[str(i)],
                                                                                                  caches_batch[str(i)])
                dx[str(i)], grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dx_batch[str(i)], caches[str(i)])
            else:
                dx[str(i)], grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dx_relu[str(i)], caches[str(i)])
                
            
 
        
            #regularizer
            grads['W'+str(i)] += self.reg * self.params['W'+str(i)]

        grads['W_last'] += self.reg * self.params['W_last']
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
