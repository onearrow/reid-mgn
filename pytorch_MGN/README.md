# Re-ID MGN（Pytorch）

## Dependencies

环境配置见`requirements.txt`文件

另，在utils文件夹下新建一个空的`__init__.py`文件

## Train

### Prepare training data

Download [Market1501](http://www.liangzheng.org/Project/project_reid.html) training data.

### Begin to train

Modify the demo.sh file and add the Market1501 directory to --datadir

run `sh demo.sh`

##  Result

In the bounding_box_test of the Market-1501 dataset, there are error labels in the ids (0194, 0196, 0373, 0538, 0574, 1060, 1174, 1366), which fundamentally affects the evaluation of the model performance, The number of images in bounding_box_test has been reduced from 19,732 to 19,698. Cleaning (others not processed), the test results are shown in the table.

In the bounding_box_train of the Market-1501 dataset, there are also some mislabelings, which are misleading to the training of the model. The relevant samples are cleaned to obtain Market-1501(C). The MGN_01_11_M_C model and the MGN_01_15_M_C_H model in the table use Market- 1501 (C) data set training, you can see a significant improvement in model performance.

![image](https://github.com/lwplw/repository_image/blob/master/%E9%80%89%E5%8C%BA_175.png)

Download model file in [here](https://drive.google.com/open?id=1SLwyC138S-wcuTBnDhYD_dzKUnqFt3nC)（MGN_12_27_M）

## Experiment

The influence of data enhancement methods RC, RF and RE on the training results of Reid MGN model.

The data enhancement in the test is used to directly process the training set image during the training process.

![image](https://github.com/lwplw/repository_image/blob/master/%E9%80%89%E5%8C%BA_171.png)

## Reference

fork of https://github.com/seathiefwang/MGN-pytorch
