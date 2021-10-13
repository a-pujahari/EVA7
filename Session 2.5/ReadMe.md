# Session 2.5 - Pytorch 101

## Assignment

Write a neural network that can:\
take 2 inputs: 
1. an image from the MNIST dataset (say 5), and 
2. a random number between 0 and 9, (say 7) 

and gives two outputs:

1. the "number" that was represented by the MNIST image (predict 5), and
2. the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)


## Data Generation Strategy

New class of type Dataset created, which under getitem() is designed to return an MNIST image, corresponding label, a random number and the sum of the random number with the corresponding label.

Random numbers are generated using torch.randint() method. Once one-hot encoded using torch.nn.functional.one_hot, it needs to be converted to datatype torch.float32 in order to prevent data type mismatches during further processing.

## Data Representation

MNIST Images - image matrices (single channel), converted to tensors and normalized based on dataset statistics (mean, std). Stacked in a batch using DataLoader for input to NN
Labels - vector containing labels for corresponding MNIST images
Random number - generated using torch.randint(), on-hot encoded and converted to datatype torch.float32

## Combining Data
MNIST Images

28x28 images are passed through 2 blocks of 5x5 kernel convolutions followed by RELU and Max-pooling each time. 

Conv layer 1

Input image 28x28x1, Output 24x24x6

Max pool Input 24x24x6, Output 12x12x6

Conv layer 2

Input 12x12x6, Output 8x8x12

Max pool Input 8x8x12, Output 4x4x12

Reshape to vector - 192 elements

Random Number

One hot encoded vector of 10 elements, is passed through a fully connected layer with 20 output features

Combining data

192 elements from MNIST convolution block are concatenated with 20 elements from fully connected random number block to create a vector of 212 elements which are then further passed through successive fully connected layers

## Results - Evaluation & Explanation

Results are evaluated after every training epoch with total number of correct predictions for the MNIST image number and the random number.

Final results are poor, with the network failing to predict either MNIST image numbers or random numbers correctly. 

Neural network is unable to learn because of random number FC output being added to MNIST convolution ouput and then being passed through a series of fully connected layers. 

Lack of a discernible pattern with the random numbers generated leads to the sum output being random as well; with the neural network having no specific pattern/characteristics/properties to learn and exploit to create a reliable prediction.

## Loss Function

Cross entropy was picked as the loss function for backpropagation and gradient descent as both outputs are currently evaluated against labels with multiple classes (>2).

10 classes for MNIST images, and 19 total classes for random number + MNIST sum. 


## Training Logs

loss=4.556783676147461 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.98it/s] 

Test set: Average MNIST loss: 16120.5077, MNIST_Accuracy: 15/10000 (0%)


Test set: Average SUM loss: 29087.6550, SUM_Accuracy: 2/10000 (0%)

loss=4.421332359313965 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.06it/s] 

Test set: Average MNIST loss: 14934.1432, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 29059.8678, SUM_Accuracy: 1/10000 (0%)

loss=4.349020004272461 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.95it/s] 

Test set: Average MNIST loss: 14856.4494, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 29067.5078, SUM_Accuracy: 1/10000 (0%)

loss=4.365970611572266 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.17it/s] 

Test set: Average MNIST loss: 14819.7078, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 28982.7053, SUM_Accuracy: 1/10000 (0%)

loss=4.3660736083984375 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.28it/s]

Test set: Average MNIST loss: 14776.3891, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 28936.4774, SUM_Accuracy: 1/10000 (0%)

loss=4.345150470733643 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.17it/s] 

Test set: Average MNIST loss: 14778.4739, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 28951.8773, SUM_Accuracy: 2/10000 (0%)

loss=4.386104106903076 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.14it/s] 

Test set: Average MNIST loss: 14757.3095, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 28922.4121, SUM_Accuracy: 4/10000 (0%)

loss=4.410125255584717 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.16it/s] 

Test set: Average MNIST loss: 14751.9032, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 28909.0461, SUM_Accuracy: 1/10000 (0%)

loss=4.383705139160156 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.01it/s] 

Test set: Average MNIST loss: 14753.7473, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 28911.3549, SUM_Accuracy: 2/10000 (0%)

loss=4.366145133972168 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.19it/s] 

Test set: Average MNIST loss: 14763.4907, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 28158.1536, SUM_Accuracy: 3/10000 (0%)

loss=4.175370216369629 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.46it/s] 

Test set: Average MNIST loss: 14757.5667, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26802.1787, SUM_Accuracy: 6/10000 (0%)

loss=4.145566463470459 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.40it/s] 

Test set: Average MNIST loss: 14760.4933, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26341.2856, SUM_Accuracy: 8/10000 (0%)

loss=4.087268829345703 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.43it/s] 

Test set: Average MNIST loss: 14773.6829, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26183.2460, SUM_Accuracy: 8/10000 (0%)

loss=4.104599952697754 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.30it/s] 

Test set: Average MNIST loss: 14791.4089, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26205.8363, SUM_Accuracy: 11/10000 (0%)

loss=4.10409688949585 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.46it/s]  

Test set: Average MNIST loss: 14770.6918, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26109.5138, SUM_Accuracy: 9/10000 (0%)

loss=4.116916656494141 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.30it/s] 

Test set: Average MNIST loss: 14763.8929, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26022.5709, SUM_Accuracy: 9/10000 (0%)

loss=3.9998509883880615 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.38it/s]

Test set: Average MNIST loss: 14732.0695, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25997.0903, SUM_Accuracy: 7/10000 (0%)

loss=4.093625068664551 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.33it/s] 

Test set: Average MNIST loss: 14763.1750, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26014.7305, SUM_Accuracy: 9/10000 (0%)

loss=4.120395183563232 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.47it/s] 

Test set: Average MNIST loss: 14740.1077, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26052.7187, SUM_Accuracy: 11/10000 (0%)

loss=4.061442852020264 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.38it/s] 

Test set: Average MNIST loss: 14770.9882, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 26063.0530, SUM_Accuracy: 3/10000 (0%)

loss=4.037917613983154 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.04it/s] 

Test set: Average MNIST loss: 14752.2930, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25909.7451, SUM_Accuracy: 8/10000 (0%)

loss=4.080263137817383 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.57it/s] 

Test set: Average MNIST loss: 14747.4512, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25852.8116, SUM_Accuracy: 8/10000 (0%)

loss=4.0878496170043945 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.24it/s]

Test set: Average MNIST loss: 14760.7230, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25801.1494, SUM_Accuracy: 8/10000 (0%)

loss=4.081490516662598 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.40it/s] 

Test set: Average MNIST loss: 14756.2902, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25849.4827, SUM_Accuracy: 10/10000 (0%)

loss=4.040626525878906 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.30it/s] 

Test set: Average MNIST loss: 14726.2158, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25799.1529, SUM_Accuracy: 10/10000 (0%)

loss=4.10402774810791 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.31it/s]  

Test set: Average MNIST loss: 14749.7999, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25888.9111, SUM_Accuracy: 7/10000 (0%)

loss=4.156913757324219 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.47it/s] 

Test set: Average MNIST loss: 14756.4907, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25755.3869, SUM_Accuracy: 9/10000 (0%)

loss=4.0890984535217285 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.47it/s]

Test set: Average MNIST loss: 14738.7848, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25882.2365, SUM_Accuracy: 7/10000 (0%)

loss=4.051886081695557 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.96it/s] 

Test set: Average MNIST loss: 14717.3454, MNIST_Accuracy: 16/10000 (0%)


Test set: Average SUM loss: 25777.6849, SUM_Accuracy: 3/10000 (0%)
