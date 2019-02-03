# mAP (mean Average Precision)
This code calculates the mAP(Mean Average Precision) of the object detection.

In general, a higher mAP value indicates a better performance of the detection, given your ground-truth and set of classes.

The performance of your detector will be judged using the mAP criterium defined in the [PASCAL VOC competition](http://host.robots.ox.ac.uk/pascal/VOC/). There are some differences with the mAP
criterium after VOC2010. The details can be found in [VOCdevkit2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar) and [VOCdevkit2012]
(http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar), which are writen in Matlab. We intepret these codes as python and support both criteriums.

## Prerequisites
- Python3 (We didn't adapt Python2 yet)
- numpy
- matplotlib (Optional)

## Run

Step by step:

1. [Create the ground-truth files](#create-the-ground-truth-files)
2. [Create the predicted objects files](#create-the-predicted-objects-files)
3. Run the code:
```shell
python mAP.py -t ground-truth -r result
```

- To use the old metric, add the flag '--use_07_metric'.
- To plot the result, add the flag '-g'.

### Create the ground-truth files

- Create a separate ground-truth text file for each image.
- Use **matching names** (e.g. image: "image_1.jpg", ground-truth: "image_1.txt"; "image_2.jpg", "image_2.txt"...).
- In these files, each line should be in the following format:
```
<class_name><left><top><right><bottom><difficult>
```
In the last column, 0 indicates a normal object and 1 indicates a difficult object, which will be ignored accroding to the PASCAL VOC document.

- E.g. "image_1.txt":
```
tvmonitor 2 10 173 238 0
book 439 157 556 241 0
book 437 246 518 351 1
pottedplant 272 190 316 259 0
```

We proivid a tool that can convert the ground-truth file from PASCAL VOC Annotations, see the [tools](https://github.com/artinfo1982/demo).

### Create the predicted objects files

- For each class, a separate '**\<class_name\>**.txt' should be genarated, e.g. 'car.txt'.
- Each line contains a record in the following format:
```
<image identifier><confidence><left><top><right><bottom>
```

- E.g. "car.txt":
```
000004 0.702732 89 112 516 466
000006 0.870849 373 168 488 229
000006 0.852346 407 157 500 213
000006 0.914587 2 161 55 221
000008 0.532489 175 184 232 201
```