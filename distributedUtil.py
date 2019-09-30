""" Utility constants, functions, and classes for distributed training of a VGG-Funnel
    convolutional neural network
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch

# ------------- CONFIGURATION VALUES -----------------------------------------------------------

BACKEND = "gloo"        # Backend for distributed training

EPOCHS = 20             # Default number of epochs
LEARNING_RATE = 0.001   # Learning rate for back-propagation
MOMENTUM = 0.9          # Momentum coefficient to smooth gradient history
DECAY_STEP_SIZE = 7
DECAY_GAMMA = 0.1

BATCH_SIZE = 16          # Data is loaded in batches during training.

DATA_DIR = {
    "t": "TrainData",
    "v": "ValidationData"
}

# Weights to address unbalanced classes. Needs to be generic for any number of classes and any class size!!
# CLASS_OPTIM_WEIGHTS = [0.41, 0.19, 0.4]


class Average(object):
    """ Aggregates and counts values entered using the update() method, with the purpose of
        computing the average.
        Methods:
            update(value, number) Adds number*values to a running sum. Increases the
            count variable to keep track of number of items
        Properties:
            average (float): Returns the average of running sum
    """

    def __init__(self):
        """ Initializes the object """
        self.sum = 0
        self.count = 0

    def __str__(self):
        """ Defines the display format of the average"""
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        """ Returns the average of the running sum (can be invoked as a method or as a property)"""
        return self.sum / self.count

    def update(self, value, number):
        """ Updates the running sum and item count """
        self.sum += value * number
        self.count += number


class Accuracy2(object):
    """ Aggregates the number of correct answers (from neural network predictions)
    and returns the average accuracy

        Methods:
            update(output, target): Takes the output tensor (the output of processing a mini batch
                and the target vector (the ground-truth values for the mini batch), and computes
                the number of correct answers.
        Properties:
            accuracy (float): Returns the average accuracy computed using the aggregated correct answers

    """

    def __init__(self):
        """ Initializes the object """
        self.correct = 0
        self.count = 0

    def __str__(self):
        """ Defines the display format of the accuracy value """
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        """ Returns the average accuracy (can be invoked as a property or as a method) """
        return self.correct / self.count

    def update(self, output, target):
        """ Updates the number of correct answers using the current output tensor (for a mini batch)
            and the true values for the mini batch.
        """
        with torch.no_grad():

            # Original code
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)
