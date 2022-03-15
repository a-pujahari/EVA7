# EVA7 - Capstone Project - Submission 1

## Submission by Abhinav Pujahari (abhinavpujahari@gmail.com)

## DETR Panoptic Segmentation Overview
![DETR_Panoptic](https://github.com/a-pujahari/EVA7/blob/main/Capstone/detr_panoptic.png)

## Questions & Answers:

### 1. Where do we take the encoded image from?
We obtain the encoded image from the output of the image encoder CNN from the object detection stage of DETR.

### 2. How do we generate attention maps?
(N x M x H/32 x W/32) attention maps are low resolution attention maps generated from the multi head attention attention module with M heads. It generates M low resolution heatmaps per N objects in the encoded image feature resolution of H/32 x W/32.

### 3. Where is the Res5 block coming from?
Res5, Res4, Res3, Res2 are feature blocks derived from the original image encoder CNN. In the case of the original paper, considering Resnet is used, these are Resnet features.

### 4. Explain the DETR Panoptic Segmentation steps.
Explanation for sequential steps of DETR Panoptic Segmentation Mask Head:
Step 1: Input box embeddings into multi head attention module with M heads to calculate attention maps over encoded image. M attention maps are output for every object (over N objects) with the same encoded image feature resolution of H/32 x W/32
Step 2: Attention maps obtained are parallely upsampled using a FPN style CNN to create mask logits for every object. This requires input of ResNet features as well.
Step 3: FPN output contains mask logits for every object, on which pixelwise argmax operation can be perfomed to obtain the final panoptic segmented output.
