import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 200,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print(event_acc.Tags())

    training_accuracies =   event_acc.Scalars('ACC/accuracy')
    #validation_accuracies = event_acc.Scalars('ACC/test')

    steps = 200
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2] # value
        #y[i, 1] = validation_accuracies[i][2]

    return x, y

def plot_(x,y):
    
    plt.plot(x, y[:,0], label='training accuracy')
    #plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='lower right', frameon=True)
    plt.show()


if __name__ == '__main__':
    log_file = "./Aug21_00-19-11_gibis-SYS-7038A-I-CIFAR10_ORIGINAL-EP200-SM0.0-A1.0-B1.0"
    x , y = plot_tensorflow_log(log_file)
    plot_(x,y)
