# TensorFlow2.0_SSD
A tensorflow_2.0 implementation of SSD (Single Shot MultiBox Detector) .

【训练的中间结果明显有问题，但是我找不到bug所在，如果你发现了bug，请开issue或者提PR，谢谢~】
(The intermediate results of the training are obviously problematic, but I can't find the bug.If you find a bug, please open an issue or raise a PR, thank you~)

## Requirements:
+ Python >= 3.8
+ TensorFlow >= 2.5.0

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
3. Run **write_voc_to_txt.py** to generate **voc.txt**.
4. Run **train.py** to start training, before that, you can change the value of the parameters in **configuration.py**.

### Test on single picture
1. Change the *test_picture_dir* in **configuration.py**.
2. Run **test.py** to test on single picture.


## References
+ The paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
+ https://github.com/amdegroot/ssd.pytorch

