import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from nn import *

train_data = scipy.io.loadmat("../data/nist36_train.mat")
valid_data = scipy.io.loadmat("../data/nist36_valid.mat")
test_data = scipy.io.loadmat("../data/nist36_test.mat")

train_x, train_y = train_data["train_data"], train_data["train_labels"]
valid_x, valid_y = valid_data["valid_data"], valid_data["valid_labels"]
test_x, test_y = test_data["test_data"], test_data["test_labels"]

if False:  # view the data
    np.random.shuffle(train_x)
    for crop in train_x:
        plt.imshow(crop.reshape(32, 32).T, cmap="Greys")
        plt.show()

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 1e-2
hidden_size = 64
##########################
##### your code here #####
##########################


batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# initialize layers
initialize_weights(train_x.shape[1], hidden_size, params, "layer1")
initialize_weights(hidden_size, train_y.shape[1], params, "output")
layer1_W_initial = np.copy(params["Wlayer1"])  # copy for Q3.3

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
for itr in range(max_iters):
    # record training and validation loss and accuracy for plotting
    h1 = forward(train_x, params, "layer1")
    probs = forward(h1, params, "output", softmax)
    loss, acc = compute_loss_and_acc(train_y, probs)
    train_loss.append(loss / train_x.shape[0])
    train_acc.append(acc)
    h1 = forward(valid_x, params, "layer1")
    probs = forward(h1, params, "output", softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss / valid_x.shape[0])
    valid_acc.append(acc)

    total_loss = 0
    avg_acc = 0
    for xb, yb in batches:
        # forward
        h1 = forward(xb, params, "layer1")
        probs = forward(h1, params, "output", softmax)

        # loss
        loss, acc = compute_loss_and_acc(yb, probs)

        # be sure to add loss and accuracy to epoch totals
        total_loss += loss
        avg_acc += acc

        delta1 = probs - yb # derivative of cross_entropy(softmax(x)) which comes out to be probs - 1(correct values)

        # backward
        delta2 = backwards(delta1, params, "output", linear_deriv)
        backwards(delta2, params, "layer1", sigmoid_deriv)

        # gradients should be summed over batch samples
        for name in ["layer1", "output"]:
            params["W" + name] -= learning_rate*params["grad_W" + name]
            params["b" + name] -= learning_rate*params["grad_b" + name]
    
    avg_acc /= batch_num

    if itr % 2 == 0:
        print(
            "itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(
                itr, total_loss, avg_acc
            )
        )

# record final training and validation accuracy and loss
h1 = forward(train_x, params, "layer1")
probs = forward(h1, params, "output", softmax)
loss, acc = compute_loss_and_acc(train_y, probs)
train_loss.append(loss / train_x.shape[0])
train_acc.append(acc)
h1 = forward(valid_x, params, "layer1")
probs = forward(h1, params, "output", softmax)
loss, acc = compute_loss_and_acc(valid_y, probs)
valid_loss.append(loss / valid_x.shape[0])
valid_acc.append(acc)

# report validation accuracy; aim for 75%
print("Validation accuracy: ", valid_acc[-1])

# compute and report test accuracy
h1 = forward(test_x, params, "layer1")
test_probs = forward(h1, params, "output", softmax)
_, test_acc = compute_loss_and_acc(test_y, test_probs)
print("Test accuracy: ", test_acc)

# save the final network
import pickle

saved_params = {k: v for k, v in params.items() if "_" not in k}
with open("q3_weights.pickle", "wb") as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot loss curves
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.plot(range(len(valid_loss)), valid_loss, label="validation")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss) - 1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()

# plot accuracy curves
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.plot(range(len(valid_acc)), valid_acc, label="validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(train_acc) - 1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()


# Q3.3

# visualize weights
fig = plt.figure(figsize=(8, 8))
plt.title("Layer 1 weights after initialization")
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
for i, ax in enumerate(grid):
    ax.imshow(layer1_W_initial[:, i].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()

v = np.max(np.abs(params["Wlayer1"]))
fig = plt.figure(figsize=(8, 8))
plt.title("Layer 1 weights after training")
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
for i, ax in enumerate(grid):
    ax.imshow(
        params["Wlayer1"][:, i].reshape((32, 32)).T, cmap="Greys", vmin=-v, vmax=v
    )
    ax.set_axis_off()
plt.show()

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1], train_y.shape[1]))
label_indices = np.argmax(test_y, axis=1)
predicted_indices = np.argmax(test_probs, axis=1)
for true_label, pred_label in zip(label_indices, predicted_indices):
    confusion_matrix[true_label, pred_label] += 1

import string

plt.imshow(confusion_matrix, interpolation="nearest")
plt.grid()
plt.xticks(
    np.arange(36), string.ascii_uppercase[:26] + "".join([str(_) for _ in range(10)])
)
plt.yticks(
    np.arange(36), string.ascii_uppercase[:26] + "".join([str(_) for _ in range(10)])
)
plt.xlabel("predicted label")
plt.ylabel("true label")
plt.colorbar()
plt.show()
