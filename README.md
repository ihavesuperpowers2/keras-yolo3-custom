# keras-yolo3

## Note from michhar

This is a fork/modification of the excellent project https://github.com/qqwweee/keras-yolo3 - see that repo for the latest updates in original codebase.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## System

This codebase has be tested with:

- CUDA 9.0
- cuDNN 7.1
- Windows 10
- NVIDIA GPU GTX 1060

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```

> For Tiny YOLOv3 download the weights from:  https://pjreddie.com/media/files/yolov3-tiny.weights

Run on video or image file:

```
usage: yolo_video.py [-h] [--model_path MODEL_PATH]
                     [--anchors_path ANCHORS_PATH]
                     [--classes_path CLASSES_PATH] [--gpu_num GPU_NUM]
                     [--image] [--input [INPUT]] [--output [OUTPUT]]

  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to model weight file, default model_data/yolo.h5
  --anchors_path ANCHORS_PATH
                        path to anchor definitions, default
                        model_data/yolo_anchors.txt
  --classes_path CLASSES_PATH
                        path to class definitions, default
                        model_data/coco_classes.txt
  --gpu_num GPU_NUM     Number of GPU to use, default 1
  --image               Image detection mode, will ignore all positional
                        arguments
  --input [INPUT]       Video input path
  --output [OUTPUT]     [Optional] Video output path
```

For Tiny YOLOv3, just do in a similar way.

---

4. MultiGPU usage is an optional. Change the number of gpu and add gpu device id.

## Data

Use the VoTT (<a href="https://github.com/Microsoft/VoTT">link</a>) labeling tool if using custom data and export to **Tensorflow Pascal VOC**.

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

* VoTT tool with export to voc format was used, then `voc_annotation.py` to get the necessary list files - https://github.com/Microsoft/VoTT

2. To convert the darknet format of weights to Keras format, make sure you have run the following using the proper config file

`python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  

  * The file model_data/yolo_weights.h5 is, next in training, used to load pretrained weights.

3. Train with `train.py` with the following script arguments:

```
usage: train.py [-h] [--model MODEL] [--gpu_num GPU_NUM]
                [--annot_path ANNOT_PATH] [--class_path CLASS_PATH]
                [--anchors_path ANCHORS_PATH]

  -h, --help            show this help message and exit
  --model MODEL         path to model weight file, default model_data/yolo.h5
  --gpu_num GPU_NUM     Number of GPU to use, default 1
  --annot_path ANNOT_PATH
                        Annotation file with image location and bboxes
  --class_path CLASS_PATH
                        Text file with class names one per line
  --anchors_path ANCHORS_PATH
                        Text file with anchor positions, comma separated, on
                        one line.
```

Note, if you want to use original pretrained weights for YOLOv3:

1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
2. rename it as darknet53.weights  
3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. Default anchors can be used. If you use your own anchors, probably some changes are needed (using `model_data/yolo_tiny_anchors.txt`).

2. The inference result is not totally the same as Darknet but the difference is small.

3. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

4. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

5. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

6. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
