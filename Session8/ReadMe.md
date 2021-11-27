# Session 8 Assignment - Advanced Training Concepts
## Late Assignment Submission by Abhinav Pujahari (abhinavpujahari@gmail.com)

## Goals
1. Modularize code further by creating individual folders for models, utilities (training, testing, helper functions etc), containing generalized callable functions.
2. Upload modularized code to GitHub - to be used for all future assignments
3. Train ResNet18 model on CIFAR10 dataset for 40 epochs:
    * Use albumentations transforms/augmentations of RandomCrop, CutOut and Rotate
    * Use ReduceLROnPlateau as learning rate scheduler
    * Use Layer Normalization
4. Show gradcam outputs for 20 misclassified images

## Torch CV Utils

As required, an additional repo was created with models, utils and main files in the following location: [Torch_CV_Utils](https://github.com/a-pujahari/Torch_CV_Utils)

## Notebook

The notebook for this assignment can be accessed here (patience=10): [EVA7_AssignmentS8_AdvancedTrainingConcepts](https://github.com/a-pujahari/EVA7/blob/main/Session8/EVA7_AssignmentS8_AdvancedTrainingConcepts.ipynb) \
And for a second attempt with patience = 2 for LR scheduler ReduceLROnPlateau: [EVA7_AssignmentS8_Patience2]()

## Analysis
Attempt 1 - Patience value = 10 \
Epochs - 40 \
Best Training Accuracy - 82.46% (39th Epoch) \
Best Testing Accuracy - 81.33% (40th Epoch) 

Optimizer - Adam (learning rate = 0.01, weight decay = 1e-5 \
Scheduler - ReduceLROnPlateau (mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08, verbose=False) \
Finding - [threshold mode "abs" is necessary for mode = "min" when loss is expected to be negative](https://github.com/pytorch/pytorch/issues/38622)

Attempt 2 - Patience value = 2 \
Epochs - 40 \
Best Training Accuracy - 88.23% (39th Epoch) \
Best Testing Accuracy - 84.79% (40th Epoch) 

Optimizer - Adam (learning rate = 0.01, weight decay = 1e-5 \
Scheduler - ReduceLROnPlateau (mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

## Loss Curves
Loss and Accuracy curves for attempt 1 with patience = 10
![Loss and Accuracy](https://github.com/a-pujahari/EVA7/blob/main/Session8/Loss%20and%20Accuracy.png)

Loss and Accuracy curves for attempt 2 with patience = 2
![Loss and Accuracy 2](https://github.com/a-pujahari/EVA7/blob/main/Session8/Loss%20and%20Accuracy%202.png)

## Sample Misclassified Images (from attempt 1)
![Misclassified](https://github.com/a-pujahari/EVA7/blob/main/Session8/misclassified.png)

## GradCam Output (on misclassified images from attempt 1)
![gradcam1](https://github.com/a-pujahari/EVA7/blob/main/Session8/gradcam1.png)
![gradcam2](https://github.com/a-pujahari/EVA7/blob/main/Session8/gradcam2.png)

## Analysis
ReduceLROnPlateau is not triggered in the first attempt(the learning rate is not reduced) considering the patience metric is not met (number of epochs with no improvement). Test loss steadily decreases  from the beginning of training and ends up fluctating around 0.004 from epochs 30-40.

For attempt 2, learning rate gradually decreases as the patience metric is intermittently triggered during training, reducing learning rate by a factor of 0.1 each time. Eventually, learning rate drops below 1e-5 which plateaus progress and testing accuracy doesn't increase beyond 84.79%.

## Training Logs (for attempt 1)
EPOCH: 1
  0%|          | 0/391 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Loss=2.012754201889038 Batch_id=390 LR=0.01000 Accuracy=19.82: 100%|██████████| 391/391 [00:40<00:00,  9.76it/s]
Test set: Average loss: 0.0150, Accuracy: 3093/10000 (30.93%)

EPOCH: 2
Loss=1.7621138095855713 Batch_id=390 LR=0.01000 Accuracy=34.63: 100%|██████████| 391/391 [00:40<00:00,  9.76it/s]
Test set: Average loss: 0.0129, Accuracy: 4083/10000 (40.83%)

EPOCH: 3
Loss=1.4728333950042725 Batch_id=390 LR=0.01000 Accuracy=41.85: 100%|██████████| 391/391 [00:40<00:00,  9.76it/s]
Test set: Average loss: 0.0117, Accuracy: 4607/10000 (46.07%)

EPOCH: 4
Loss=1.4317330121994019 Batch_id=390 LR=0.01000 Accuracy=46.19: 100%|██████████| 391/391 [00:39<00:00,  9.79it/s]
Test set: Average loss: 0.0105, Accuracy: 5209/10000 (52.09%)

EPOCH: 5
Loss=1.4276669025421143 Batch_id=390 LR=0.01000 Accuracy=50.38: 100%|██████████| 391/391 [00:39<00:00,  9.78it/s]
Test set: Average loss: 0.0097, Accuracy: 5551/10000 (55.51%)

EPOCH: 6
Loss=1.2379728555679321 Batch_id=390 LR=0.01000 Accuracy=54.49: 100%|██████████| 391/391 [00:39<00:00,  9.79it/s]
Test set: Average loss: 0.0092, Accuracy: 5824/10000 (58.24%)

EPOCH: 7
Loss=1.371418833732605 Batch_id=390 LR=0.01000 Accuracy=57.83: 100%|██████████| 391/391 [00:39<00:00,  9.79it/s]
Test set: Average loss: 0.0085, Accuracy: 6105/10000 (61.05%)

EPOCH: 8
Loss=1.083987832069397 Batch_id=390 LR=0.01000 Accuracy=60.78: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0081, Accuracy: 6390/10000 (63.90%)

EPOCH: 9
Loss=1.165813684463501 Batch_id=390 LR=0.01000 Accuracy=63.51: 100%|██████████| 391/391 [00:39<00:00,  9.78it/s]
Test set: Average loss: 0.0074, Accuracy: 6697/10000 (66.97%)

EPOCH: 10
Loss=1.226562261581421 Batch_id=390 LR=0.01000 Accuracy=66.06: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0075, Accuracy: 6622/10000 (66.22%)

EPOCH: 11
Loss=0.7053403258323669 Batch_id=390 LR=0.01000 Accuracy=67.64: 100%|██████████| 391/391 [00:39<00:00,  9.79it/s]
Test set: Average loss: 0.0069, Accuracy: 6909/10000 (69.09%)

EPOCH: 12
Loss=0.7233192920684814 Batch_id=390 LR=0.01000 Accuracy=69.47: 100%|██████████| 391/391 [00:40<00:00,  9.75it/s]
Test set: Average loss: 0.0066, Accuracy: 7052/10000 (70.52%)

EPOCH: 13
Loss=0.742613673210144 Batch_id=390 LR=0.01000 Accuracy=70.25: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0067, Accuracy: 7124/10000 (71.24%)

EPOCH: 14
Loss=0.6871010661125183 Batch_id=390 LR=0.01000 Accuracy=71.69: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0063, Accuracy: 7234/10000 (72.34%)

EPOCH: 15
Loss=0.6745272874832153 Batch_id=390 LR=0.01000 Accuracy=72.43: 100%|██████████| 391/391 [00:40<00:00,  9.76it/s]
Test set: Average loss: 0.0064, Accuracy: 7167/10000 (71.67%)

EPOCH: 16
Loss=0.7032912969589233 Batch_id=390 LR=0.01000 Accuracy=73.30: 100%|██████████| 391/391 [00:40<00:00,  9.76it/s]
Test set: Average loss: 0.0059, Accuracy: 7350/10000 (73.50%)

EPOCH: 17
Loss=0.9277316927909851 Batch_id=390 LR=0.01000 Accuracy=74.18: 100%|██████████| 391/391 [00:39<00:00,  9.79it/s]
Test set: Average loss: 0.0057, Accuracy: 7516/10000 (75.16%)

EPOCH: 18
Loss=0.47708043456077576 Batch_id=390 LR=0.01000 Accuracy=74.77: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0057, Accuracy: 7522/10000 (75.22%)

EPOCH: 19
Loss=0.6535903811454773 Batch_id=390 LR=0.01000 Accuracy=75.60: 100%|██████████| 391/391 [00:40<00:00,  9.76it/s]
Test set: Average loss: 0.0054, Accuracy: 7637/10000 (76.37%)

EPOCH: 20
Loss=0.5895441770553589 Batch_id=390 LR=0.01000 Accuracy=76.70: 100%|██████████| 391/391 [00:40<00:00,  9.62it/s]
Test set: Average loss: 0.0056, Accuracy: 7609/10000 (76.09%)

EPOCH: 21
Loss=0.6654236912727356 Batch_id=390 LR=0.01000 Accuracy=76.96: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0051, Accuracy: 7741/10000 (77.41%)

EPOCH: 22
Loss=0.5769477486610413 Batch_id=390 LR=0.01000 Accuracy=77.53: 100%|██████████| 391/391 [00:40<00:00,  9.77it/s]
Test set: Average loss: 0.0051, Accuracy: 7771/10000 (77.71%)

EPOCH: 23
Loss=0.6968601942062378 Batch_id=390 LR=0.01000 Accuracy=78.37: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0049, Accuracy: 7815/10000 (78.15%)

EPOCH: 24
Loss=0.7165046334266663 Batch_id=390 LR=0.01000 Accuracy=78.33: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0050, Accuracy: 7784/10000 (77.84%)

EPOCH: 25
Loss=0.6491035223007202 Batch_id=390 LR=0.01000 Accuracy=78.81: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0047, Accuracy: 7892/10000 (78.92%)

EPOCH: 26
Loss=0.5965710282325745 Batch_id=390 LR=0.01000 Accuracy=79.32: 100%|██████████| 391/391 [00:39<00:00,  9.78it/s]
Test set: Average loss: 0.0049, Accuracy: 7876/10000 (78.76%)

EPOCH: 27
Loss=0.4801393449306488 Batch_id=390 LR=0.01000 Accuracy=79.55: 100%|██████████| 391/391 [00:40<00:00,  9.75it/s]
Test set: Average loss: 0.0046, Accuracy: 7960/10000 (79.60%)

EPOCH: 28
Loss=0.7023158073425293 Batch_id=390 LR=0.01000 Accuracy=79.92: 100%|██████████| 391/391 [00:40<00:00,  9.76it/s]
Test set: Average loss: 0.0049, Accuracy: 7887/10000 (78.87%)

EPOCH: 29
Loss=0.5678344368934631 Batch_id=390 LR=0.01000 Accuracy=80.23: 100%|██████████| 391/391 [00:40<00:00,  9.77it/s]
Test set: Average loss: 0.0046, Accuracy: 7983/10000 (79.83%)

EPOCH: 30
Loss=0.6629022359848022 Batch_id=390 LR=0.01000 Accuracy=80.46: 100%|██████████| 391/391 [00:40<00:00,  9.75it/s]
Test set: Average loss: 0.0046, Accuracy: 8038/10000 (80.38%)

EPOCH: 31
Loss=0.3855358064174652 Batch_id=390 LR=0.01000 Accuracy=80.69: 100%|██████████| 391/391 [00:40<00:00,  9.77it/s]
Test set: Average loss: 0.0046, Accuracy: 8005/10000 (80.05%)

EPOCH: 32
Loss=0.4624019265174866 Batch_id=390 LR=0.01000 Accuracy=81.12: 100%|██████████| 391/391 [00:39<00:00,  9.78it/s]
Test set: Average loss: 0.0045, Accuracy: 8093/10000 (80.93%)

EPOCH: 33
Loss=0.4627779424190521 Batch_id=390 LR=0.01000 Accuracy=81.34: 100%|██████████| 391/391 [00:40<00:00,  9.74it/s]
Test set: Average loss: 0.0045, Accuracy: 8067/10000 (80.67%)

EPOCH: 34
Loss=0.5073493719100952 Batch_id=390 LR=0.01000 Accuracy=81.78: 100%|██████████| 391/391 [00:40<00:00,  9.77it/s]
Test set: Average loss: 0.0047, Accuracy: 7997/10000 (79.97%)

EPOCH: 35
Loss=0.5263336300849915 Batch_id=390 LR=0.01000 Accuracy=81.64: 100%|██████████| 391/391 [00:40<00:00,  9.77it/s]
Test set: Average loss: 0.0045, Accuracy: 8079/10000 (80.79%)

EPOCH: 36
Loss=0.5256760120391846 Batch_id=390 LR=0.01000 Accuracy=81.76: 100%|██████████| 391/391 [00:40<00:00,  9.77it/s]
Test set: Average loss: 0.0045, Accuracy: 8059/10000 (80.59%)

EPOCH: 37
Loss=0.5812170505523682 Batch_id=390 LR=0.01000 Accuracy=81.94: 100%|██████████| 391/391 [00:40<00:00,  9.73it/s]
Test set: Average loss: 0.0047, Accuracy: 8064/10000 (80.64%)

EPOCH: 38
Loss=0.5160831212997437 Batch_id=390 LR=0.01000 Accuracy=82.00: 100%|██████████| 391/391 [00:39<00:00,  9.79it/s]
Test set: Average loss: 0.0044, Accuracy: 8081/10000 (80.81%)

EPOCH: 39
Loss=0.6047196388244629 Batch_id=390 LR=0.01000 Accuracy=82.46: 100%|██████████| 391/391 [00:40<00:00,  9.77it/s]
Test set: Average loss: 0.0043, Accuracy: 8124/10000 (81.24%)

EPOCH: 40
Loss=0.3570351004600525 Batch_id=390 LR=0.01000 Accuracy=82.44: 100%|██████████| 391/391 [00:39<00:00,  9.78it/s]
Test set: Average loss: 0.0044, Accuracy: 8133/10000 (81.33%)


