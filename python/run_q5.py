import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from util import relu, relu_deriv
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
momentum = 0.9

# Get random batches
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024, 32, params, name="layer0")
initialize_weights(32, 32, params, name="layer1")
initialize_weights(32, 32, params, name="layer2")
initialize_weights(32, 1024, params, name="output")

# Initialize momentum values
for name in ["layer0", "layer1", "layer2", "output"]:
    params["m_W" + name] = np.zeros_like(params["W"+name])
    params["m_b" + name] = np.zeros_like(params["b"+name])


# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # forward pass
        h1 = forward(xb, params, "layer0", relu)
        h2 = forward(h1, params, "layer1", relu)
        h3 = forward(h2, params, "layer2", relu)
        probs = forward(h3, params, "output", sigmoid)

        # your loss is now squared error
        loss = np.sum((probs-xb)**2)
        total_loss += loss

        # delta is the d/dx of (x-y)^2
        delta = 2*(probs-xb)

        # backward
        delta1 = backwards(delta, params, "output", sigmoid_deriv)
        delta2 = backwards(delta1, params, "layer2", relu_deriv)
        delta3 = backwards(delta2, params, "layer1", relu_deriv)
        delta4 = backwards(delta3, params, "layer0", relu_deriv)

        # to implement momentum
        for name in ["layer0", "layer1", "layer2", "output"]:
            params["m_W" + name] = momentum*params["m_W" + name] - learning_rate*params["grad_W" + name]
            params["m_b" + name] = momentum*params["m_b" + name] - learning_rate*params["grad_b" + name]
            params["W" + name] += params["m_W"+name]
            params["b" + name] += params["m_b"+name]
        
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
h1 = forward(visualize_x, params, "layer0", relu)
h2 = forward(h1, params, "layer1", relu)
h3 = forward(h2, params, "layer2", relu)
reconstructed_x = forward(h3, params, "output", sigmoid)

# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
psnr = np.sum(20 * np.log10(np.max(reconstructed_x, axis=1)) - 10 * np.log10(np.square(reconstructed_x).mean()))
print(psnr)
