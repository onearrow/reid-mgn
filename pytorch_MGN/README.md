# Re-ID MGN模型 Pytorch框架下的训练
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

| date | mAP | rank1 | rank3 | rank5 | rank10 |
| :------: | :------: | :------: | :------: | :------: | :------: |
|2018.12.04| 94.18 | 95.87 | 97.48 | 97.98 | 98.43 |
|2018.12.11| 94.72 | 96.14 | 97.80 | 98.16 | 98.78 |
|2018.12.12| 94.68 | 96.26 | 97.57 | 97.89 | 98.60 |

Download model file in [here](https://drive.google.com/open?id=1gb3iViwOS0dxvjM_X7O1LNafpA2iH9qS)（lwp-2018.12.11）

## Reference

fork of https://github.com/seathiefwang/MGN-pytorch
