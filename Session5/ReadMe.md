# EVA7 - Session 5 - Coding Drill Down Assignment - Submission by Abhinav Pujahari (abhinavpujahari@gmail.com), Group 9

Target - Achieve 99.4% testing accuracy consistently by optimizing a model over multiple steps for the MNIST dataset.

## Steps

### 1. Model framework (High number of parameters)
Set up working code with basic data transformations, data loaders, model (taken from 8th step of Session 5), test and train functions

### 2. Model skeleton (model parameter reduction)
Reduce number of overall parameters to below 10k, by reducing number of channels in expansions after image input and post max-pooling.

### 3. Adding Batch Normalization
Batch normalization improves contrast across all channels in every convolution layer,  improves rate of convergence during training leading to higher accuracy in the same limited number of epochs.

Batch normalization is not added to the last layer.

### 4. Adding Dropout
Dropout helps regularize the network and reduce overfitting. Reduces the gap between training and test accuracy.
Dropout is also not added to the final layer.

### 5. Adding Data Augmentations
Data augmentation (random rotation, random affine transformations and color jitter) in this case help improve testing accuracy by helping the NN adapt to different variations in the dataset.
Also adds an additional regularization effect.

### 6. Adding Learning Rate Scheduling
Learning rate scheduling helps to optimize the process of stochastic gradient descent to help achieve required target accuracy within 15 epochs. Lambda LR scheduling is used.

## Final Model
![FinalModel](https://github.com/a-pujahari/EVA7/blob/main/Session5/FinalModel.png)

## Receptive Field Calculations

![ReceptiveField](https://github.com/a-pujahari/EVA7/blob/main/Session5/receptiveField.png)

| Operation   | nin | in\_ch | out\_ch | padding | kernel | stride | nout | jin | jout | rin | rout | 
| ----------- | --- | ------ | ------- | ------- | ------ | ------ | ---- | --- | ---- | --- | ---- | 
| Convolution | 28  | 1      | 8       | 0       | 3      | 1      | 26   | 1   | 1    | 1   | 3    |
| Convolution | 26  | 8      | 16      | 0       | 3      | 1      | 24   | 1   | 1    | 3   | 5    |
| Max-Pooling | 24  | 16     | 16      | 0       | 2      | 2      | 12   | 1   | 2    | 5   | 6    |
| Convolution | 12  | 16     | 8       | 0       | 1      | 1      | 12   | 2   | 2    | 6   | 6    |
| Convolution | 12  | 8      | 16      | 0       | 3      | 1      | 10   | 2   | 2    | 6   | 10   |
| Convolution | 10  | 16     | 24      | 0       | 3      | 1      | 8    | 2   | 2    | 10  | 14   |
| GAP         | 8   | 24     | 24      | 0       | 8      | 1      | 1    | 2   | 2    | 14  | 28   |
| Convolution | 1   | 24     | 32      | 0       | 1      | 1      | 1    | 2   | 2    | 28  | 28   |
| Convolution | 1   | 32     | 16      | 0       | 1      | 1      | 1    | 2   | 2    | 28  | 28   |
| Convolution | 1   | 16     | 10      | 0       | 1      | 1      | 1    | 2   | 2    | 28  | 28   |

## Training Logs (For Final Step)

EPOCH: 1
Loss=0.4880262017250061 Batch_id=937 Accuracy=84.46: 100%|██████████| 938/938 [01:39<00:00,  9.41it/s]

Test set: Average loss: 0.0890, Accuracy: 9707/10000 (97.07%)

EPOCH: 2
Loss=0.24310088157653809 Batch_id=937 Accuracy=94.74: 100%|██████████| 938/938 [01:36<00:00,  9.76it/s]

Test set: Average loss: 0.0397, Accuracy: 9869/10000 (98.69%)

EPOCH: 3
Loss=0.12208434194326401 Batch_id=937 Accuracy=95.74: 100%|██████████| 938/938 [01:34<00:00,  9.88it/s]

Test set: Average loss: 0.0454, Accuracy: 9859/10000 (98.59%)

EPOCH: 4
Loss=0.49917522072792053 Batch_id=937 Accuracy=96.36: 100%|██████████| 938/938 [01:35<00:00,  9.83it/s]

Test set: Average loss: 0.0336, Accuracy: 9896/10000 (98.96%)

EPOCH: 5
Loss=0.16808158159255981 Batch_id=937 Accuracy=96.57: 100%|██████████| 938/938 [01:34<00:00,  9.92it/s]

Test set: Average loss: 0.0269, Accuracy: 9903/10000 (99.03%)

EPOCH: 6
Loss=0.13319484889507294 Batch_id=937 Accuracy=96.80: 100%|██████████| 938/938 [01:38<00:00,  9.53it/s]

Test set: Average loss: 0.0256, Accuracy: 9916/10000 (99.16%)

EPOCH: 7
Loss=0.04289092496037483 Batch_id=937 Accuracy=96.92: 100%|██████████| 938/938 [01:36<00:00,  9.71it/s]

Test set: Average loss: 0.0277, Accuracy: 9909/10000 (99.09%)

EPOCH: 8
Loss=0.15397752821445465 Batch_id=937 Accuracy=97.19: 100%|██████████| 938/938 [01:37<00:00,  9.66it/s]

Test set: Average loss: 0.0209, Accuracy: 9936/10000 (99.36%)

EPOCH: 9
Loss=0.14810331165790558 Batch_id=937 Accuracy=97.23: 100%|██████████| 938/938 [01:37<00:00,  9.66it/s]

Test set: Average loss: 0.0232, Accuracy: 9932/10000 (99.32%)

EPOCH: 10
Loss=0.006525579374283552 Batch_id=937 Accuracy=97.21: 100%|██████████| 938/938 [01:35<00:00,  9.79it/s]

Test set: Average loss: 0.0208, Accuracy: 9932/10000 (99.32%)

EPOCH: 11
Loss=0.0516677051782608 Batch_id=937 Accuracy=97.34: 100%|██████████| 938/938 [01:36<00:00,  9.71it/s]

Test set: Average loss: 0.0191, Accuracy: 9948/10000 (99.48%)

EPOCH: 12
Loss=0.3774683177471161 Batch_id=937 Accuracy=97.40: 100%|██████████| 938/938 [01:36<00:00,  9.72it/s]

Test set: Average loss: 0.0175, Accuracy: 9941/10000 (99.41%)

EPOCH: 13
Loss=0.022144075483083725 Batch_id=937 Accuracy=97.39: 100%|██████████| 938/938 [01:36<00:00,  9.77it/s]

Test set: Average loss: 0.0185, Accuracy: 9945/10000 (99.45%)

EPOCH: 14
Loss=0.07377509772777557 Batch_id=937 Accuracy=97.59: 100%|██████████| 938/938 [01:37<00:00,  9.66it/s]

Test set: Average loss: 0.0209, Accuracy: 9941/10000 (99.41%)

EPOCH: 15
Loss=0.03509518876671791 Batch_id=937 Accuracy=97.60: 100%|██████████| 938/938 [01:36<00:00,  9.68it/s]

Test set: Average loss: 0.0184, Accuracy: 9942/10000 (99.42%)
