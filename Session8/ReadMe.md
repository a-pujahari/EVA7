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

The notebook for this assignment can be accessed here: [EVA7_AssignmentS8_AdvancedTrainingConcepts]()

## Analysis
Epochs - 40
Best Training Accuracy - 84.00% (40th Epoch)
Best Testing Accuracy - 83.33% (37th Epoch)

Optimizer - Adam (learning rate = 0.01, weight decay = 1e-5

Scheduler - ReduceLROnPlateau (mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

Finding - [threshold mode "abs" is necessary for mode = "min" when loss is expected to be negative](https://github.com/pytorch/pytorch/issues/38622)

## Loss Curves


## GradCam Output (on misclassified images)

## Analysis
ReduceLROnPlateau is not triggered (the learning rate is not reduced) considering the patience metric is not met (number of epochs with no improvement). Test loss steadily decreases  from the beginning of training and ends up fluctating around 0.004 from epochs 30-40.




