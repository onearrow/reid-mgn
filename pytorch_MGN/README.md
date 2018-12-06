# re-id_mgn Pytorch框架下的训练
Reproduction of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

## Dependencies

环境配置见requirements.txt文件

## Train

### Prepare training data

Download Market1501 training data.[here](http://www.liangzheng.org/Project/project_reid.html)

### Begin to train

Modify the demo.sh file and add the Market1501 directory to --datadir

run `sh demo.sh`

##  Result

|  | mAP | rank1 | rank3 | rank5 | rank10 |
| :------: | :------: | :------: | :------: | :------: | :------: |
|lwp-2018.12.5| 94.18 | 95.87 | 97.48 | 97.98 | 98.43 |

Download model file in [here](https://pan.baidu.com/s/1DbZsT16yIITTkmjRW1ifWQ)

## Reference

fork of https://github.com/seathiefwang/MGN-pytorch
