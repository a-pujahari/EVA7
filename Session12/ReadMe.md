# EVA7 - Session 12 - Dawn of Transformers - Assignment
## Submission by Abhinav Pujahari (abhinavpujahari@gmail.com)

# Spatial Transformer Task
## Tasks:
1. Implement Spatial Transformer code for CIFAR10
2. Train for 50 Epochs, show sample outputs

## Notebooks
Link to the notebook on github can be found [here](https://github.com/a-pujahari/EVA7/blob/main/Session12/EVA7_Session12_SpatialTransformer.ipynb)
The public notebook on Colab can be found [here](https://colab.research.google.com/drive/1_Ebvu7ZRWKOfOGZpQPQLP1YMdufoO0jy?usp=sharing)

## Spatial Transformer

Spatial transformer network allows the neural network to learn to perform spatial transformations on the input image to reduce the overall geometric variance of the model.
This is helpful considering neural networks are not completely invariant to scale, rotation and affine transformations. 

![spatial_transformer](https://github.com/a-pujahari/EVA7/blob/main/Session12/stn-arch.png)

There are 3 main components to the spatial transformer network:
1. Localization network - CNN which regresses the transformation parameters. Parameters are not learned explicitly from the dataset, but through optimization of global accuracy during training.
2. Grid generator - generates a grid of coordinates in the input image corresponding to each pixel from the output image.
3. Sampler - applies the parameters of the transformation to the input image.

## Sample CIFAR10 Output Images
![Sample_output](https://github.com/a-pujahari/EVA7/blob/main/Session12/CIFAR10_SpatialTransformer.png)
Sample outputs show how input images have undergone affine transformations to center objects and equalize size across objects of different classes. 

## Training Logs
Train Epoch: 1 [0/50000 (0%)]	Loss: 2.301728
Train Epoch: 1 [32000/50000 (64%)]	Loss: 1.981925

Test set: Average loss: 1.8304, Accuracy: 3431/10000 (34%)

Train Epoch: 2 [0/50000 (0%)]	Loss: 2.004818
Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.682925

Test set: Average loss: 1.6111, Accuracy: 4202/10000 (42%)

Train Epoch: 3 [0/50000 (0%)]	Loss: 1.792376
Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.718828

Test set: Average loss: 1.5415, Accuracy: 4474/10000 (45%)

Train Epoch: 4 [0/50000 (0%)]	Loss: 1.819720
Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.573076

Test set: Average loss: 1.4425, Accuracy: 4832/10000 (48%)

Train Epoch: 5 [0/50000 (0%)]	Loss: 1.329195
Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.455489

Test set: Average loss: 1.4184, Accuracy: 4880/10000 (49%)

Train Epoch: 6 [0/50000 (0%)]	Loss: 1.430230
Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.434268

Test set: Average loss: 1.3516, Accuracy: 5149/10000 (51%)

Train Epoch: 7 [0/50000 (0%)]	Loss: 1.355284
Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.534765

Test set: Average loss: 1.3464, Accuracy: 5206/10000 (52%)

Train Epoch: 8 [0/50000 (0%)]	Loss: 1.295988
Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.210028

Test set: Average loss: 1.3852, Accuracy: 5143/10000 (51%)

Train Epoch: 9 [0/50000 (0%)]	Loss: 1.701890
Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.127555

Test set: Average loss: 1.2415, Accuracy: 5662/10000 (57%)

Train Epoch: 10 [0/50000 (0%)]	Loss: 1.455071
Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.424353

Test set: Average loss: 1.2629, Accuracy: 5607/10000 (56%)

Train Epoch: 11 [0/50000 (0%)]	Loss: 1.313301
Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.253022

Test set: Average loss: 1.3660, Accuracy: 5196/10000 (52%)

Train Epoch: 12 [0/50000 (0%)]	Loss: 1.547763
Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.254721

Test set: Average loss: 1.2152, Accuracy: 5667/10000 (57%)

Train Epoch: 13 [0/50000 (0%)]	Loss: 1.406668
Train Epoch: 13 [32000/50000 (64%)]	Loss: 1.206839

Test set: Average loss: 1.1735, Accuracy: 5889/10000 (59%)

Train Epoch: 14 [0/50000 (0%)]	Loss: 1.023594
Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.364591

Test set: Average loss: 1.2183, Accuracy: 5730/10000 (57%)

Train Epoch: 15 [0/50000 (0%)]	Loss: 1.132396
Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.291381

Test set: Average loss: 1.1585, Accuracy: 5945/10000 (59%)

Train Epoch: 16 [0/50000 (0%)]	Loss: 1.095431
Train Epoch: 16 [32000/50000 (64%)]	Loss: 1.049565

Test set: Average loss: 1.1647, Accuracy: 5936/10000 (59%)

Train Epoch: 17 [0/50000 (0%)]	Loss: 1.356097
Train Epoch: 17 [32000/50000 (64%)]	Loss: 1.146808

Test set: Average loss: 1.0946, Accuracy: 6251/10000 (63%)

Train Epoch: 18 [0/50000 (0%)]	Loss: 1.068451
Train Epoch: 18 [32000/50000 (64%)]	Loss: 0.824458

Test set: Average loss: 1.1344, Accuracy: 6031/10000 (60%)

Train Epoch: 19 [0/50000 (0%)]	Loss: 1.050137
Train Epoch: 19 [32000/50000 (64%)]	Loss: 1.109381

Test set: Average loss: 1.1586, Accuracy: 5943/10000 (59%)

Train Epoch: 20 [0/50000 (0%)]	Loss: 1.256548
Train Epoch: 20 [32000/50000 (64%)]	Loss: 1.160666

Test set: Average loss: 1.1424, Accuracy: 6089/10000 (61%)

Train Epoch: 21 [0/50000 (0%)]	Loss: 1.329542
Train Epoch: 21 [32000/50000 (64%)]	Loss: 1.022500

Test set: Average loss: 1.0775, Accuracy: 6256/10000 (63%)

Train Epoch: 22 [0/50000 (0%)]	Loss: 1.363867
Train Epoch: 22 [32000/50000 (64%)]	Loss: 1.311817

Test set: Average loss: 1.0805, Accuracy: 6307/10000 (63%)

Train Epoch: 23 [0/50000 (0%)]	Loss: 1.097380
Train Epoch: 23 [32000/50000 (64%)]	Loss: 1.017828

Test set: Average loss: 1.0743, Accuracy: 6342/10000 (63%)

Train Epoch: 24 [0/50000 (0%)]	Loss: 1.037147
Train Epoch: 24 [32000/50000 (64%)]	Loss: 0.875904

Test set: Average loss: 1.0463, Accuracy: 6489/10000 (65%)

Train Epoch: 25 [0/50000 (0%)]	Loss: 1.146446
Train Epoch: 25 [32000/50000 (64%)]	Loss: 1.082771

Test set: Average loss: 1.0540, Accuracy: 6359/10000 (64%)

Train Epoch: 26 [0/50000 (0%)]	Loss: 1.043418
Train Epoch: 26 [32000/50000 (64%)]	Loss: 1.103940

Test set: Average loss: 1.0386, Accuracy: 6462/10000 (65%)

Train Epoch: 27 [0/50000 (0%)]	Loss: 1.088127
Train Epoch: 27 [32000/50000 (64%)]	Loss: 1.241538

Test set: Average loss: 1.0240, Accuracy: 6537/10000 (65%)

Train Epoch: 28 [0/50000 (0%)]	Loss: 0.966950
Train Epoch: 28 [32000/50000 (64%)]	Loss: 0.880452

Test set: Average loss: 1.0225, Accuracy: 6571/10000 (66%)

Train Epoch: 29 [0/50000 (0%)]	Loss: 0.983755
Train Epoch: 29 [32000/50000 (64%)]	Loss: 1.143795

Test set: Average loss: 1.0662, Accuracy: 6301/10000 (63%)

Train Epoch: 30 [0/50000 (0%)]	Loss: 0.918912
Train Epoch: 30 [32000/50000 (64%)]	Loss: 0.964199

Test set: Average loss: 1.0152, Accuracy: 6534/10000 (65%)

Train Epoch: 31 [0/50000 (0%)]	Loss: 1.220566
Train Epoch: 31 [32000/50000 (64%)]	Loss: 0.833259

Test set: Average loss: 1.0154, Accuracy: 6510/10000 (65%)

Train Epoch: 32 [0/50000 (0%)]	Loss: 1.029564
Train Epoch: 32 [32000/50000 (64%)]	Loss: 0.801605

Test set: Average loss: 1.0323, Accuracy: 6484/10000 (65%)

Train Epoch: 33 [0/50000 (0%)]	Loss: 1.037148
Train Epoch: 33 [32000/50000 (64%)]	Loss: 0.903843

Test set: Average loss: 1.0335, Accuracy: 6434/10000 (64%)

Train Epoch: 34 [0/50000 (0%)]	Loss: 1.126167
Train Epoch: 34 [32000/50000 (64%)]	Loss: 0.968298

Test set: Average loss: 1.0561, Accuracy: 6400/10000 (64%)

Train Epoch: 35 [0/50000 (0%)]	Loss: 0.633604
Train Epoch: 35 [32000/50000 (64%)]	Loss: 0.724081

Test set: Average loss: 1.0165, Accuracy: 6604/10000 (66%)

Train Epoch: 36 [0/50000 (0%)]	Loss: 0.971604
Train Epoch: 36 [32000/50000 (64%)]	Loss: 0.732642

Test set: Average loss: 1.0204, Accuracy: 6569/10000 (66%)

Train Epoch: 37 [0/50000 (0%)]	Loss: 0.777975
Train Epoch: 37 [32000/50000 (64%)]	Loss: 0.557565

Test set: Average loss: 1.0126, Accuracy: 6608/10000 (66%)

Train Epoch: 38 [0/50000 (0%)]	Loss: 0.770925
Train Epoch: 38 [32000/50000 (64%)]	Loss: 1.110042

Test set: Average loss: 1.0463, Accuracy: 6457/10000 (65%)

Train Epoch: 39 [0/50000 (0%)]	Loss: 0.701298
Train Epoch: 39 [32000/50000 (64%)]	Loss: 0.889331

Test set: Average loss: 1.0015, Accuracy: 6628/10000 (66%)

Train Epoch: 40 [0/50000 (0%)]	Loss: 0.951702
Train Epoch: 40 [32000/50000 (64%)]	Loss: 0.847342

Test set: Average loss: 0.9926, Accuracy: 6654/10000 (67%)

Train Epoch: 41 [0/50000 (0%)]	Loss: 0.728503
Train Epoch: 41 [32000/50000 (64%)]	Loss: 0.817686

Test set: Average loss: 1.0943, Accuracy: 6257/10000 (63%)

Train Epoch: 42 [0/50000 (0%)]	Loss: 0.763408
Train Epoch: 42 [32000/50000 (64%)]	Loss: 0.575945

Test set: Average loss: 1.0573, Accuracy: 6449/10000 (64%)

Train Epoch: 43 [0/50000 (0%)]	Loss: 0.778861
Train Epoch: 43 [32000/50000 (64%)]	Loss: 0.699535

Test set: Average loss: 1.1882, Accuracy: 6193/10000 (62%)

Train Epoch: 44 [0/50000 (0%)]	Loss: 1.209074
Train Epoch: 44 [32000/50000 (64%)]	Loss: 0.707676

Test set: Average loss: 1.1008, Accuracy: 6268/10000 (63%)

Train Epoch: 45 [0/50000 (0%)]	Loss: 0.759709
Train Epoch: 45 [32000/50000 (64%)]	Loss: 0.581050

Test set: Average loss: 0.9905, Accuracy: 6699/10000 (67%)

Train Epoch: 46 [0/50000 (0%)]	Loss: 0.795403
Train Epoch: 46 [32000/50000 (64%)]	Loss: 0.782734

Test set: Average loss: 1.0043, Accuracy: 6655/10000 (67%)

Train Epoch: 47 [0/50000 (0%)]	Loss: 0.666026
Train Epoch: 47 [32000/50000 (64%)]	Loss: 0.740014

Test set: Average loss: 1.0910, Accuracy: 6370/10000 (64%)

Train Epoch: 48 [0/50000 (0%)]	Loss: 0.579505
Train Epoch: 48 [32000/50000 (64%)]	Loss: 0.697954

Test set: Average loss: 1.1973, Accuracy: 6095/10000 (61%)

Train Epoch: 49 [0/50000 (0%)]	Loss: 0.873521
Train Epoch: 49 [32000/50000 (64%)]	Loss: 0.565028

Test set: Average loss: 1.0591, Accuracy: 6469/10000 (65%)

Train Epoch: 50 [0/50000 (0%)]	Loss: 0.605282
Train Epoch: 50 [32000/50000 (64%)]	Loss: 0.644525

Test set: Average loss: 1.0898, Accuracy: 6415/10000 (64%)

The overall output accuracy does not improve beyond ~65%, this could be due to a limitation of the base CNN.
