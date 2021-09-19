# Assignment 0

## Submission by Abhinav Pujahari, Group 9 - Other members include Yuvaraj V and Satwik S Mishra

### 1. What are Channels & Kernels according to EVA?
Channels are fundamental building blocks of an image or other data (text etc), that can be used in various combinations to recreate every sample of the data set. I.e straight lines, circles, curves for image data, RGB channels for image data, individual characters or set of words for text data etc.
Kernels are convolutional filters/matrices that help break down data into these fundamental building blocks.

### 2. Why should we (nearly) always use 3x3 kernels?
We should always use 3x3 kernels for the following reasons:
1. Computational efficiency - using a 3x3 will reduce the overall number of parameters (as against using larger kernels)
2. Support for hardware acceleration for 3x3 kernel convolutions from GPU manufacturers
3. Easier to keep track of number of parameters and image sizes through multiple layers (with stride of 1 and padding of 0)
4. 3x3 kernels can help to preserve symmetry

### 3. How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
We will need to perform 98 3x3 convolutions to reach a matrix of size 1x1 from 199x199.
199x199 > 197x197 > 195x195 > 193x193 > 191x191 > 189x189 > 187x187 > 185x185 > 183x183 > 181x181 >
179x179 > 177x177 > 175x175 > 173x173 > 171x171 > 169x169 > 167x167 > 165x165 > 163x163 > 161x161 >
159x159 > 157x157 > 155x155 > 153x153 > 151x151 > 149x149 > 147x147 > 145x145 > 143x143 > 141x141 >
139x139 > 137x137 > 135x135 > 133x133 > 131x131 > 129x129 > 127x127 > 125x125 > 123x123 > 121x121 >
119x119 > 117x117 > 115x115 > 113x113 > 111x111 > 109x109 > 107x107 > 105x105 > 103x103 > 101x101 >
99x99 > 97x97 > 95x95 > 93x93 > 91x91 > 89x89 > 87x87 > 85x85 > 83x83 > 81x81 >
79x79 > 77x77 > 75x75 > 73x73 > 71x71 > 69x69 > 67x67 > 65x65 > 63x63 > 61x61 >
59x59 > 57x57 > 55x55 > 53x53 > 51x51 > 49x49 > 47x47 > 45x45 > 43x43 > 41x41 >
39x39 > 37x37 > 35x35 > 33x33 > 31x31 > 29x29 > 27x27 > 25x25 > 23x23 > 21x21 >
19x19 > 17x17 > 15x15 > 13x13 > 11x11 > 9x9 > 7x7 > 5x5 > 3x3 > 1x1


### 4. How are kernels initialized? 
Kernels are initialized randomly. Initializing kernels with presumed knowledge, i.e creating filters to break down images based on human intution or known information (templates to detct edges, shapes, objects etc) may not be optimized for the dataset in question. Initiating kernels randomly allows for the network weights, i.e kernel values to be optimized through training to a point where they can find features that lead to maximum discrimination between dissimilar examples in the dataset.

### 5. What happens during the training of a DNN?
Training allows for network weights, i.e kernel values to be optimized to a point where features in the dataset that provide more discrimination/separation ability across individual data examples can be recognized and used by the DNN for the required task (classification, object detection etc).

