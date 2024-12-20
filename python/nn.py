import numpy as np
from util import *

# do not include any more libraries here!
# do not put any code outside of functions!


############################## Q 2.1.2 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]
def initialize_weights(in_size, out_size, params, name=""):
    W, b = None, None
    lower_bound = (-1*np.sqrt(6))/(np.sqrt(in_size + out_size))
    upper_bound = (np.sqrt(6))/(np.sqrt(in_size + out_size))
    W = np.random.uniform(lower_bound, upper_bound, [in_size, out_size])
    b = np.zeros(out_size)

    params["W" + name] = W
    params["b" + name] = b


############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res


############################## Q 2.2.1 ##############################
def forward(X, params, name="", activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params["W" + name]
    b = params["b" + name]

    pre_act = X @ W + b
    post_act = activation(pre_act)
    
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params["cache_" + name] = (X, pre_act, post_act)

    return post_act


############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None
    max_elements = np.max(x, axis=1, keepdims=True)
    x = x - max_elements
    res = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return res


############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    # Loss
    loss = -np.sum(y*np.log(probs))
    
    # Accuracy
    maximum_probs_indices = np.argmax(probs, axis=1)
    y_at_max = y[np.arange(0, len(y)), maximum_probs_indices]
    acc = np.sum(y_at_max) / len(y_at_max)

    return loss, acc


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act * (1.0 - post_act)
    return res


def backwards(delta, params, name="", activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params["W" + name]
    b = params["b" + name]
    X, pre_act, post_act = params["cache_" + name]

    del_f_del_a = activation_deriv(post_act)
    del_a_del_w = X # N x d
    del_a_del_x = W # d x k

    # Compute dZ
    dZ = delta * del_f_del_a
    
    # Compute gradients
    grad_W = del_a_del_w.T @ dZ
    grad_b = np.sum(dZ, axis=0)
    grad_X = dZ @ del_a_del_x.T
    

    # store the gradients
    params["grad_W" + name] = grad_W
    params["grad_b" + name] = grad_b
    return grad_X


############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x, y, batch_size):
    # Same indices for batches_i_x and batches_i_y
    # Batch_size is given
    # Must handle unequal batch size for the last batch
    batches = []
    assert len(x) == len(y) # We want x and y to be the same sizes
    indices = np.arange(0, len(x)) # From 0 to len(x) - 1
    np.random.shuffle(indices) # Shuffles indices in place
    for i in range(0, len(indices)//batch_size):
        batch_indices = indices[i*batch_size : (i+1)*batch_size] #0 to 10, 10 to 20 ... of shuffled indices
        batch_x = x[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_x, batch_y)) # Construct a batch len(indices)//batch_size times
    return batches
