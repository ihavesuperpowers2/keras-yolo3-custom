# Keras implementation of YOLO v3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

YOLO stands for you only look once and is an efficient algorithm for object detection.

![YOLO image](https://cdn-images-1.medium.com/max/1600/1*QOGcvHbrDZiCqTG6THIQ_w.png)
<a href="https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088" target="_blank" align="right">Image Source</a>

Important papers on YOLO:

* Original - https://arxiv.org/abs/1506.02640
* 9000/v2 - https://arxiv.org/abs/1612.08242
* v3 - https://arxiv.org/abs/1804.02767

There are "tiny" versions of the architecture, often considered for embedded/constrained devices.

Website:  https://pjreddie.com/darknet/yolo/ (provides information on a framework called Darknet)

This implementation of YOLOv3 (Tensorflow backend) was inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

---

## Updates Provided in This Fork

* Option to **bring custom data and labels**
* Parameterization of scripts
* More detailed instructions and introductory material (above)

## What You Do Here

Using this repo you will perform some or all of the following:

* Convert a Darknet model to a Keras model (and if custom setup, modify the config file with proper filter and class numbers)
* Perform inference on video or image as a test ([quick-start](#quick-start))
* Label data with a bounding box definition tool
* Train a model on custom data using the converted custom Darknet model in Keras format (`.h5`)
* Perform inference on custom model

## System

This codebase has be tested with:

- CUDA 9.0
- cuDNN 7.1
- Windows 10
- NVIDIA GPU GTX 1060
- Anaconda Python 3.6

## Quick Start (Inference Only)

1. `pip install requirements.txt`
2. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
3. Convert the Darknet YOLO model to a Keras model.
4. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py experiment/yolov3.cfg yolov3.weights model_data/yolo.h5
```

> For tiny YOLOv3 download the weights from:  https://pjreddie.com/media/files/yolov3-tiny.weights

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

e.g.  `python yolo_video.py --model_path model_data/yolo.h5 --anchors model_data/yolo_anchors.txt --classes_path model_data/coco_classes.txt`

> For Tiny YOLOv3, just do in a similar way, except with tiny YOLOv3, converted weights.

---

4. MultiGPU usage is an optional. Change the number of gpu and add gpu device id.

## Data Prep

Use the VoTT (<a href="https://github.com/Microsoft/VoTT">link</a>) labeling tool if using custom data and export to **Tensorflow Pascal VOC**.

## Training

IMPORTANT NOTES:

* Make sure you have set up the config `.cfg` file correctly (`filters` and `classes`) - more information on how to do this <a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects" target="_blank">here</a>
* Make sure you have converted the weights by running:  `python convert.py yolov3-custom-for-project.cfg yolov3.weights model_data/yolo-custom-for-project.h5` (i.e. for config update the `filters` in CNN layer above `[yolo]`s and `classes` in `[yolo]`'s to class number)

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example of the output:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

* To get the necessary annotation files `voc_annotation.py` was used and then the files to be used for train were combined into one file, e.g. as in the `example_label_list.txt`.

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
