# Re-ID MGN（Pytorch）
Reproduction of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

## Dependencies

环境配置见`requirements.txt`文件

另，在utils文件夹下新建一个空的`__init__.py`文件

## Train

### Prepare training data

Download Market1501 training data.[here](http://www.liangzheng.org/Project/project_reid.html)

### Begin to train

Modify the demo.sh file and add the Market1501 directory to --datadir

run `sh demo.sh`

##  Result
![image](https://github.com/lwplw/repository_image/blob/master/%E9%80%89%E5%8C%BA_170.png)

Download model file in [here](https://drive.google.com/open?id=1OG37yTbLVgPMi1N1aDySyhJMp5kWMHBm)（lwp-2018.12.11）

## Experiment

The influence of data enhancement methods RC, RF and RE on the training results of Reid MGN model.

The data enhancement in the test is used to directly process the training set image during the training process.

![image](https://github.com/lwplw/repository_image/blob/master/%E9%80%89%E5%8C%BA_171.png)

## Reference

fork of https://github.com/seathiefwang/MGN-pytorch
