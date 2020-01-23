# TensorFlow2.0_SSD
A tensorflow_2.0 implementation of SSD (Single Shot MultiBox Detector) .

**Note: This project is under development, which means it is not available now.**

## Requirements:
+ Python >= 3.6
+ TensorFlow >= 2.0.0-rc1
+ Pillow >= 6.1.0

## Usage
### Train on PASCAL VOC 2012
1. Download the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
2. Unzip the file and place it in the 'dataset' folder, make sure the directory is like this : 
```
|——dataset
    |——VOCdevkit
        |——VOC2012
            |——Annotations
            |——ImageSets
            |——JPEGImages
            |——SegmentationClass
            |——SegmentationObject
```
3. Run **train.py** to start training, before that, you can change the value of the parameters in **configuration.py**.


## References
+ The paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
+ http://zh.gluon.ai/chapter_computer-vision/ssd.html
+ https://blog.csdn.net/XiangJiaoJun_/article/details/84503224
+ MXNet tutorial on bilibili : <br/> (1) https://www.bilibili.com/video/av16012497<br/> (2) https://www.bilibili.com/video/av16225415<br/> (3) https://www.bilibili.com/video/av16440968
