# Distributed VGG-F

Sample code showing how to run distributed training for a
VGG convolutional neural network using PyTorch Distributed
Data Parallael module. The code has been tested with 
virtual machines in the cloud, each machine having one GPU. 

VGG-F stands for VGG-Funnel. It is a VGG-16 convolutional neural net 
with two additional layers that progressively reduce the 
feature space 
from 4096 to a few number of classes (instead of 
the thousand classes required by the ImageNet design). 
For example, the referenced dataset used for testing 
the implementation has been designed to have only 3 classes. 
Having a reduced
number of classes is a much more common scenario in 
industry applications.

## Requirements

- Python 3.6
- PyTorch 1.2

## Installation

- Prepare a cluster of machines, each with a single GPU. 
- Copy code files to each of the machines in the cluster
- Copy all data files to each of the machines in the cluster. 
Note: if you are running in the cloud, you may want to save the data
to some centralized storage and modify the current 
code to load mini batches 
from there (this feature has not been implemented yet)

## Running

- If the number of machines is N (a value known as the world size), then 
each machine is identified by a number between 0 and N-1 (known as 
its rank). 

- Select one machine as the central node that aggregates gradients, The IP address of 
this machine must be passed in the `init_url` parameter

- In each machine run:

```bash
python distributedVggf.py -iu <tcp-url> -rn <rank> -ws <world-size> -rd <root-dir> -ep 5 -lr 0.00001 -mb 64
```

where 

`<tcp-url>` is the main node address expressed as a TCP URL. For example: 
`tcp://10.128.0.8`

`<rank>` is the machine's identifying number (a number between 0 and N-1)

`<world-size>` is the number N of machines in the cluster

`<root-dir>` is the data root directory. This directory hosts the  `TrainData` and 
`ValidationData` folders as described in the next section. 

Note: The learning rate and mini batch size suggested above have been 
selected to generate good results using the example dataset and classification 
problem described below. 

## Dataset File Structure

The code uses PyTorch `ImageFolder` class to access and load data. 
Hence, the file structure matches the requirements of this class. 

- There is a root directory referenced above as `<root-dir>` 
that contains two folders called 
`TrainData`  and `ValidationData`.

- Each of these folders has C sub-directories, where C is the number 
of classes. 

- The name of each sub-directory is the class label. Each 
sub-directory contains images labeled to belong to the 
respective class.

## Evaluation dataset

The collection of images known as COIL-100 was used to perform initial 
evaluation. The collection is available from:

[http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)

This dataset was prepared by the Center for Research on Intelligent Systems, Dept. 
of Computer Science at Columbia University. It contains 
color images for 100 objects taken from different angles using a rotating 
table. The camera was fixed and captured images every 5 degrees. Hence, 
the total number of images per object is 72, and the total number of 
images in the dataset is 7200. Each image is of size 128 x 128 pixels. 

For evaluation, we divided the COIL-100 dataset in three classes: edible items, 
toys, and other. The objects selected for the edible and toy categories are:

edible = {2, 4, 7, 47, 49, 53, 62, 63, 67, 72, 73, 75, 82, 83, 84, 93, 94, 98}

toy = {6, 8, 14, 15, 17, 19, 20, 23, 27, 28, 34, 37, 48, 51, 52, 69, 74, 76, 
91, 100}

Consequently, the dataset consisted of 1296 images in the `edible` class, 4464 
images in the `other` class, and 1440 images in the `toy` class. The training 
procedure used 80% of the data. The remaining 20% was used exclusively for validation. 

## Results

Two identical virtual machines running Linux were configured in Google Cloud for testing. Each 
machine included 2 virtual CPUs with 13 GB of memory, 100 GB standard persistent 
disk, 1 Tesla K80 GPU. The software stack included PyTorch 1.2 and CUDA 10.0.

The code described in this repository configures a VGG-F neural network initialized 
using the ImageNet weights. With this initialization, the VGG-F neural net can 
converge quickly into a reasonable prediction accuracy when trained with the 
3-class classification task mentioned above. It takes 2 or 3 epochs to reach 
a validation accuracy of 98% when using a small learning rate of 1e-5. 

For speed benchmarking the training algorithm run for 5 epochs using the same 
learning rate and different mini-batch sizes. 

The Distributed Data Parallel 
module in PyTorch assigns different mini batches to different instances/virtual machines. Each 
instance runs forward propagation and gradient calculation. One of the instances 
aggregates and averages the gradients. The average gradient is re-distributed 
to all the virtual machines to update the model coefficients. The process then 
continues with the next group of mini batches. 

The table below shows the results of running the training algorithm using 1 and 
2 virtual machines (vm) with different mini-batch sizes. The accuracy columns 
represent validation accuracy (i.e. using the validation data).

| mini-batch | epochs | 1 vm (time) | 1 vm (acc) | 2 vm (time) | 2 vm (acc) |
| --- | --- | --- | --- | --- | --- |
| 16 | 5 | 18m 25s | 98.05% | 23m 55s | 98.33% |
| 32 | 5 | 16m 20s | 98.19% | 14m 20s | 98.12% |
| 64 | 5 | 15m 38s | 98.26% | 9m 48s  | 98.33% |
| 96 | 5 | 21m 15s | 98.54% | 13m 51s | 98.54% |

After processing each mini batch, the individual instances send their computed 
 gradients to a central node and wait for the average of gradients. Consequently, if 
the mini-batch size is small, there will be too many of these gradient synchronization 
procedures happening over a network, which makes the overall training less 
efficient. In the above example, this happens with a size of 16 images where 
parallel training is slower than single-machine training. 

Each time a mini batch is processed, the batched images are transferred 
from storage to the GPU RAM. Hence, if the mini-batch size is large, there is a 
point where the size of the transferred images surpasses the storage limit of the 
GPU. In the example above, this condition happened with a size of 128 images.

The table above shows that the sweet spot for this particular 3-class classification 
problem is a mini-batch size of 64 images. This value results in the best 
training time when using 2 distributed virtual machines. 


## References

This work is an extension of initial work described in

[https://github.com/narumiruna/pytorch-distributed-example](https://github.com/narumiruna/pytorch-distributed-example)

In particular, this work extends the previous implementation in two directions: 
- Support for a convolutional neural network (VGG)
- Support for loading data from generic user-defined directories

PyTorch has multiple pages explaining the Distributed Data Parallel 
and related modules:

[https://pytorch.org/docs/stable/nn.html#distributeddataparallel](https://pytorch.org/docs/stable/nn.html#distributeddataparallel)



## License
[MIT](https://choosealicense.com/licenses/mit/)



