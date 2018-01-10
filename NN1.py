
# Import the relevant components
from __future__ import print_function
import numpy as np
import cntk as C
import cntk.tests.test_utils
import matplotlib.pyplot as plt
from collections import defaultdict
import math
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix the random seed so that LR examples are repeatable
# Ensure that we always get the same results
np.random.seed(0)

# Helper function to generate a random data sample
def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)

    # Specify the data type to match the input variable used later in the tutorial
    # (default type is double)
    X = X.astype(np.float32)

    # convert class 0 into the vector "1 0 0",
    # class 1 into the vector "0 1 0", ...
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

# compute the moving average.
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# Print the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error

def Plot_data():
    plt.figure(1)
    colors = ['r' if label == 0 else 'b' for label in labels[:, 0]]
    plt.scatter(features[:, 0], features[:, 1], c=colors)
    plt.xlabel("Age (scaled)")
    plt.ylabel("Tumor size (in cm)")
    plt.show(block = False)

def plotCurrentModel(bias_vector, weight_matrix):
    plt.plot([0, bias_vector[0] / weight_matrix[0][1]],
             [bias_vector[1] / weight_matrix[0][0], 0], c='g', lw=3)

def PlotTrainingError():
    global fig, y1, y2, li1, li2
    plot_data_size = int(num_minibatches_to_train / training_progress_output_freq)
    fig = plt.figure(2)
    ax1 = plt.subplot(211)
    ax1.set_ylim([0, 1.5])
    ax2 = plt.subplot(212)
    ax2.set_ylim([0, 0.6])
    # some X and Y data
    x = training_progress_output_freq * np.arange(plot_data_size)
    y1 = [2] * plot_data_size
    y2 = [2] * plot_data_size
    plt.xlim(xmax=num_minibatches_to_train)
    li1, = ax1.plot(x, y1, 'b--')
    ax1.set_ylabel('Loss')
    ax1.set_title('Minibatch run vs. Training loss/Label Prediction Error')
    plt.show(block=False)
    li2, = ax2.plot(x, y2, 'r--')
    plt.ylabel('Label Prediction Error')
    plt.xlabel('Minibatch number')
    plt.show(block=False)

def UpdateTrainingErrorPlot():
    idx = math.floor(i / training_progress_output_freq)
    y1[idx] = moving_average(plotdata["loss"])[idx]
    li1.set_ydata(y1)
    y2[idx] = moving_average(plotdata["error"])[idx]
    li2.set_ydata(y2)
    fig.canvas.draw()

def PlotTestDataResultsAndModel(bias_vector, weight_matrix):
    global label
    plt.figure(3)
    colors = ['r' if label == 0 else 'b' for label in labels[:, 0]]
    plt.scatter(features[:, 0], features[:, 1], c=colors)
    plt.plot([0, bias_vector[0] / weight_matrix[0][1]],
             [bias_vector[1] / weight_matrix[0][0], 0], c='g', lw=3)
    plt.xlabel("Patient age (scaled)")
    plt.ylabel("Tumor size (in cm)")
    plt.show()

def PlotTrainingDataAndModel():
    plt.cla()
    Plot_data()
    plotCurrentModel(mydict['b'].value, mydict['w'].value)
    plt.draw()
    plt.waitforbuttonpress()

def linear_layer(input_var, output_dim):
    mydict = {}
    input_dim = input_var.shape[0]
    weight_param = C.parameter(shape=(input_dim, output_dim))
    bias_param = C.parameter(shape=(output_dim))
    mydict['w'], mydict['b'] = weight_param, bias_param
    return C.times(input_var, weight_param) + bias_param, mydict


# Define the network
input_dim = 2
num_output_classes = 2

feature = C.input_variable(input_dim, np.float32)
label = C.input_variable(num_output_classes, np.float32)

z, mydict = linear_layer(feature, num_output_classes)
loss = C.cross_entropy_with_softmax(z, label)
eval_error = C.classification_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate = 0.5
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])

# Initialize the parameters for the trainer
minibatch_size = 25
num_samples_to_train = 20000
num_minibatches_to_train = int(num_samples_to_train  / minibatch_size)

# Run the trainer and perform model training
training_progress_output_freq = 50
plotdata = defaultdict(list)

PlotTrainingError()

for i in range(0, num_minibatches_to_train):
    features, labels = generate_random_data_sample(minibatch_size, input_dim, num_output_classes)

    # Assign the minibatch data to the input variables and train the model on the minibatch
    trainer.train_minibatch({feature : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i,
                                                     training_progress_output_freq, verbose=1)

    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)
        UpdateTrainingErrorPlot()
        # PlotTrainingDataAndModel()

# Run the trained model on a newly generated dataset
test_minibatch_size = 25
features, labels = generate_random_data_sample(test_minibatch_size, input_dim, num_output_classes)

trainer.test_minibatch({feature : features, label : labels})

PlotTestDataResultsAndModel(mydict['b'].value, mydict['w'].value)