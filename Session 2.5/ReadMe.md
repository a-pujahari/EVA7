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
