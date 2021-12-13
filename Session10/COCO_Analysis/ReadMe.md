# COCO Annotation Analysis
## Submission by Abhinav Pujahari (abhinavpujahari@gmail.com)

COCO stands for Common Objects in COntext - microsoft's dataset for object detection. The dataset's popularity has made the annotation format used extremely popular and a go-to choice for object detection tasks.

Major keys within the COCO annotation format include:
1. Info - contains high level information about the dataset
2. Licenses - licensing information for the dataset in question
3. Images - image names, urls, information such as width and height, with the corresponding class ID
4. Annotations - list of object annotations, can include segmentation data, BBox coordinates, ID etc.
5. Categories - list of categories/classes mapped to numeric IDs

## Class Distribution
![classDistribution](https://github.com/a-pujahari/EVA7/blob/main/Session10/COCO_Analysis/COCO_Class_Distribution.png)

## Anchor Boxes - K-Means Analysis
Anchor boxes are reference/template bounding boxes with the most likely size/aspect ratio suited to the objects/annotations within the dataset.

Please note that the anchor boxes plotted below are scaled and zero centered - with the intention of being representative of the relative sizes and the aspect ratios.

![Anchor3](https://github.com/a-pujahari/EVA7/blob/main/Session10/COCO_Analysis/Anchor3.png)
![Anchor4](https://github.com/a-pujahari/EVA7/blob/main/Session10/COCO_Analysis/Anchor4.png)
![Anchor5](https://github.com/a-pujahari/EVA7/blob/main/Session10/COCO_Analysis/Anchor5.png)
![Anchor6](https://github.com/a-pujahari/EVA7/blob/main/Session10/COCO_Analysis/Anchor6.png)

### K-Means Elbow Method
The Elbow method suggests k = 2 or 3 would be the ideal number of clusters for this dataset

![Elbow](https://github.com/a-pujahari/EVA7/blob/main/Session10/COCO_Analysis/ElbowMethod.png)
