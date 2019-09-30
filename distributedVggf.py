""" Distributed training using multiple instances of a convolutional neural network called VGG Funnel (VGG-F).
    A VGG Funnel network is simply  a VGG-16 with additional layers that reduce progressively the
    number of features with the goal of having a reduced number of classes as the output.

    The VGG-F network in this program is initialized using ImageNet-derived weights.

    The code described here is an extension of the original project described in:
         https://github.com/narumiruna/pytorch-distributed-example/blob/master/mnist/main.py

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import argparse

import torch
import torch.nn.functional as nnfun
from torch import distributed, nn
from torch.utils import data
from torchvision import datasets, transforms, models

import distributedUtil as dstUt


def distributed_is_initialized():
    """ Checks if a distributed cluster has been initialized """
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def vgg_funnel_model(number_classes):
    """ Returns a pre-trained VGG16 model with two extra layers to narrow down class selection to
        a few number of classes.
        Args:
           number_classes (int): the number of classes in a class
        Returns:
            A VGG model (PyTorch object) with a 2-layer extension initialized with ImageNet
            coefficients.
    """

    # Bring in the pre-trained model
    model = models.vgg16(pretrained=True)

    last_layer_input_size = model.classifier[6].in_features   # For VGG16, this layer has 4096 inputs

    # Re-define the last layer of the network (sixth layer) replacing it with a 'funnel', which is
    # a 3-layer fully connected network that progressively reduces the number of classes to a few
    model.classifier[6] = nn.Sequential(
        nn.Linear(last_layer_input_size, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(512, number_classes),
    )

    return model


# ----------------- DATA MANAGER CLASS -------------------------------------------------------
class DataManager(object):
    """ Defines all necessary functions for loading and transforming the training and validation data,
        which is stored in pre-determined folders.

        Attributes:
            class_names (list): list of class names or labels
            number_classes (int): the number of classes or labels
            data_size (int): number of images in the data set
        Methods:
            get_loader(): Returns a handler to a function that loads data in mini batches
    """

    def __init__(self, root_folder, mini_batch, train=True):
        """ Initializes the class
            Args:
                root_folder (str): The path of a root directory containing a TrainData and
                    ValidationData subdirectories.
                mini-batch (int): size of a mini batch
                train (boolean): If True, the object is used for training. Otherwise for validation.
        """
        self.root_folder = root_folder    # Root folder for the train and validation sets
        self.mb_size = mini_batch

        if train:
            self.path = os.path.join(self.root_folder, dstUt.DATA_DIR['t'])
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # ImageNet standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet st
            ])
            self.dataset = datasets.ImageFolder(self.path, self.transforms)
            self.class_names = self.dataset.classes
            self.number_classes = len(self.class_names)
            self.data_size = len(self.dataset)

        else:
            self.path = os.path.join(self.root_folder, dstUt.DATA_DIR['v'])
            self.transforms = transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet st
            ])
            self.dataset = datasets.ImageFolder(self.path, self.transforms)
            self.class_names = self.dataset.classes
            self.number_classes = len(self.class_names)
            self.data_size = len(self.dataset)

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.DistributedSampler(self.dataset)

        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.mb_size,
                                                  shuffle=(sampler is None), sampler=sampler)

    def get_loader(self):
        """ Returns a handler (PyTorch object) to a function that loads data in mini batches"""
        return self.loader


class Trainer(object):
    """ Trains a VGG-F convolutional neural network

        Methods:
            fit(epochs): Runs the training and evaluation procedures for a number of epochs

    """
    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.__train()
            test_loss, test_acc = self.__evaluate()

            print(
                '[Info] Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
            )

    def __train(self):
        """ Trains the neural network (forward and backward propagation) """
        self.model.train()

        train_loss = dstUt.Average()
        train_acc = dstUt.Accuracy2()

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)

            # code to add weights for unbalanced data
            # weight_ten = torch.tensor(dstUt.CLASS_OPTIM_WEIGHTS)
            # loss = nnfun.cross_entropy(outputs, targets, weight=weight_ten.to(self.device))

            loss = nnfun.cross_entropy(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(outputs, targets)

        return train_loss, train_acc

    def __evaluate(self):
        """ Evaluates the neural network using separate validation data """
        self.model.eval()

        test_loss = dstUt.Average()
        test_acc = dstUt.Accuracy2()

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = nnfun.cross_entropy(outputs, targets)

                test_loss.update(loss.item(), inputs.size(0))
                test_acc.update(outputs, targets)

        return test_loss, test_acc


def manage_training(args):
    """ Controls the process of training a convolutional neural network
    Args:
        args (parser object): command-line arguments selected by the user
    Returns:
        Nothing
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    train_data_mngr = DataManager(root_folder=args.root_dir, mini_batch=args.mini_batch, train=True)
    train_loader = train_data_mngr.get_loader()

    valid_data_mngr = DataManager(root_folder=args.root_dir, mini_batch=args.mini_batch, train=False)
    valid_loader = valid_data_mngr.get_loader()

    num_classes = train_data_mngr.number_classes
    print("[Info] number of classes: {}".format(num_classes))
    print("[Info] class labels: {}".format(train_data_mngr.class_names))
    print("[Info] Running instance {} using {}".format(args.rank, torch.cuda.get_device_name(device)))

    model = vgg_funnel_model(num_classes)

    if distributed_is_initialized():
        print("[Info] distributed training has been initialized")
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        model = nn.DataParallel(model)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train_loader = MNISTDataLoader(args.root, args.batch_size, train=True)
    # test_loader = MNISTDataLoader(args.root, args.batch_size, train=False)

    trainer = Trainer(model, optimizer, train_loader, valid_loader, device)
    trainer.fit(args.epochs)


def parse_command_line():
    """ Parses command-line call and extracts arguments
        Returns:
            parser object with selected arguments
    """

    desc = "Trains a VGG-F convolutional neural network using a distributed cluster of machines. "
    desc += "Train and validation data must be stored in folders TrainData and ValidadionData under some "
    desc += "root directory. Train and validation folders must have N subdirectories, one per class. "
    desc += "The name of each of these N subdirectories is the class label."
    parser = argparse.ArgumentParser(description=desc)
    required = parser.add_argument_group("required arguments")

    # ------------------ REQUIRED ARGUMENTS ----------------------------------------------------------------------
    help_iu = "Initialization URL specifying the protocol and connection with a master "
    help_iu += "node like tcp://192.168.0.10:23456"
    required.add_argument('-iu', '--init_url', type=str, help=help_iu, required=True)

    help_rn = "Rank of the current instance. If there are K instances, the rank identifies individual instances "
    help_rn += "with a number between 0 and K-1."
    required.add_argument('-rn', '--rank', type=int, default=0, help=help_rn, required=True)

    help_ws = "Number of instances participating in the job"
    required.add_argument('-ws', '--world_size', type=int, help=help_ws, required=True)

    help_rd = "Root directory for the training and validation sets"
    required.add_argument('-rd', '--root_dir', help=help_rd, type=str)

    # ------------------- OPTIONAL ARGUMENTS ----------------------------------------------------------------------
    help_e = "Number of epochs. The default value is {}".format(dstUt.EPOCHS)
    parser.add_argument('-ep', '--epochs', help=help_e, type=int, default=dstUt.EPOCHS)

    help_nc = "Flag to use the CPU in this instance. By default the program tries to run on a GPU"
    parser.add_argument('-nc', '--no_cuda', help=help_nc, action='store_true')

    help_lr = "Selected learning rate. The default value is {}".format(dstUt.LEARNING_RATE)
    parser.add_argument('-lr', '--learning_rate', help=help_lr, type=float, default=dstUt.LEARNING_RATE)

    help_mb = "Mini-batch size. The default value is {}".format(dstUt.BATCH_SIZE)
    parser.add_argument('-mb', '--mini_batch', help=help_mb, type=int, default=dstUt.BATCH_SIZE)

    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        distributed.init_process_group(
            backend=dstUt.BACKEND,
            init_method=args.init_url,
            world_size=args.world_size,
            rank=args.rank
        )

    return args


if __name__ == '__main__':
    # Set OS environment variable for Gloo
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'

    # Parse command line and extract parameters
    selected_args = parse_command_line()

    # Run training and perform validation
    manage_training(selected_args)
