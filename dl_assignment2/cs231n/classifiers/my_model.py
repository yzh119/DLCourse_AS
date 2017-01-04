import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim = ([3, 32, 32, 64, 64], 32, 32), num_filters = [32, 32, 64, 64], filter_size = 5,
               hidden_dim = 64, num_classes = 10, weight_scale = [1e-4, 1e-2, 1e-2, 1e-1, 1e-1], reg = 0.0,
               dropout = 0, dtype = np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim
    F = num_filters
    HH = WW = filter_size
    
    for i in range(3):
        self.params['W1' + repr(i)] = weight_scale[i] * \
            np.random.randn(F[i], C[i], HH, WW)
        self.params['b1' + repr(i)] = weight_scale[i] * \
            np.random.randn(F[i])
        self.params['gamma' + repr(i)] = np.ones(F[i])
        self.params['beta' + repr(i)] = np.zeros(F[i])

    self.params['W2'] = weight_scale[3] * np.random.randn(F[3] * 3 * 3, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    
    self.params['W3'] = weight_scale[4] * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    
    self.bn_params = [{'mode': 'train'} for i in xrange(3)]
    self.dropout_param = {'mode': 'train', 'p': dropout}
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    self.dropout_param['mode'] = mode
    
    scores = None
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    kernel = [4, 3, 3]
    
    out_1, cache_1, out_norm, cache_norm, W1, b1 = [None] * 3, [None] * 3, [None] * 3, [None] * 3, [None] * 3, [None] * 3
    
    for i in range(3):
        W1[i], b1[i] = self.params['W1' + repr(i)], self.params['b1' + repr(i)]
        gamma, beta = self.params['gamma' + repr(i)], self.params['beta' + repr(i)]
        filter_size = W1[i].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        pool_param = {'pool_height': kernel[i], 'pool_width': kernel[i], 'stride': 2}

        out_1[i], cache_1[i] = conv_relu_pool_forward(X, W1[i], b1[i], conv_param, pool_param)
        out_norm[i], cache_norm[i] = spatial_batchnorm_forward(out_1[i], gamma, beta, self.bn_params[i])
        X = out_norm[i]
        
    out_2, cache_2 = affine_relu_forward(X, W2, b2)            
    out_drop, cache_drop = dropout_forward(out_2, self.dropout_param)
    out_3, cache_3 = affine_forward(out_drop, W3, b3)
    scores = out_3
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dscores = softmax_loss(scores, y)
    
    dout_3, grads['W3'], grads['b3'] = affine_backward(dscores, cache_3)
    dout_drop = dropout_backward(dout_3, cache_drop)
    dout_2, grads['W2'], grads['b2'] = affine_relu_backward(dout_drop, cache_2)
    
    loss += .5 * self.reg * (np.sum(W2 ** 2) + np.sum(W3 ** 2))
    grads['W3'] += self.reg * W3
    grads['W2'] += self.reg * W2

    dout_1 = [None] * 3
    dout_cache = [None] * 3
    dout = dout_2
    for i in range(2, -1, -1):
        dout_cache[i], grads['gamma' + repr(i)], grads['beta' + repr(i)] = spatial_batchnorm_backward(dout, cache_norm[i])
        dout_1[i], grads['W1' + repr(i)], grads['b1' + repr(i)] = conv_relu_pool_backward(dout_cache[i], cache_1[i])
        loss += .5 * self.reg * np.sum(W1[i] ** 2)
        grads['W1' + repr(i)] += self.reg * W1[i]
        dout = dout_1[i]
    
    return loss, grads
  
  
pass
