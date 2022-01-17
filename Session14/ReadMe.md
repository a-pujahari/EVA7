# EVA7 - Session 14 - DETR - End to End Object Detection with Transformers

## Submission by Abhinav Pujahari (abhinavpujahari@gmail.com)

Goals:
1. Replicate the process of fine-tuning DETR on a custom dataset for object detection
2. Understand architectural concepts and how fine tuning works

## DETR
DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel.

The main objective behind DETR is effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode prior knowledge about the task and makes the process complex and computationally expensive.

![DETR_Arch](https://github.com/a-pujahari/EVA7/blob/main/Session14/DETR_Arch.png)

### DETR Architecture

DETR consists of 3 main components:
1. CNN Backbone for embedding image patches
2. Encoder-Decoder transformer
3. Feed forward network as head a prediction head

![DETR_backbone](https://github.com/a-pujahari/EVA7/blob/main/Session14/DETR_transformer.png)

### Bipartite Matching Loss

![bipartite](https://github.com/a-pujahari/EVA7/blob/main/Session14/bipartite%20matching%20loss.png)

DETR makes a fixed set of predictions which need to be narrowed down to the specific number of objects present in the image. This is done through bipartite matching i.e, one to one comparisons of the object predictions helping to eliminate low quality predictions and helping to achieve output reductions like NMS.

## Fine Tuning DETR on Custom Dataset

The notebook for the fine tuning exercise can be found [here](https://github.com/a-pujahari/EVA7/blob/main/Session14/EVA7_Session14_FineTuningDETR.ipynb).

### Metrics

Illustrated below are the loss and mean average precision value progression captured during training. 

![metrics](https://github.com/a-pujahari/EVA7/blob/main/Session14/train_metrics.png)

### Results 

![train_image](https://github.com/a-pujahari/EVA7/blob/main/Session14/val_image_result.png)

![validation_image](https://github.com/a-pujahari/EVA7/blob/main/Session14/train_image_result.png)


