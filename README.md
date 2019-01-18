# Re-ID MGN模型
Reproduction of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

## 1. Model conversion

### -[pytorch2caffe_MGN](https://github.com/lwplw/re-id_mgn/tree/master/pytorch2caffe_MGN)
- [x] Structure file conversion
- [x] Weight file conversion
- [x] Verify prediction results

### -[caffe2darknet_MGN](https://github.com/lwplw/re-id_mgn/tree/master/caffe2darknet_MGN)
- [x] Structure file conversion
- [x] Weight file conversion
- [x] Verify prediction results

### -[darknet_MGN](https://github.com/lwplw/re-id_mgn/tree/master/darknet_MGN)

DarkNet after modifying the source code.

#### （1）在caffe2darknet的转换中，slice层由于DarkNet不支持，需要修改DarkNet源码添加新层实现

#### （2）需要修改源码让max_pool层支持多尺度kernel size和stride size

#### （3）修改DarkNet源码，增加了Python接口调用对numpy格式输入的支持

## 2. Model training

### -[pytorch_MGN](https://github.com/lwplw/re-id_mgn/tree/master/pytorch_MGN)

Train the MGN model under Pytorch.


## 3. Re-id matching module
### -[matching](https://github.com/lwplw/re-id_mgn/tree/master/matching)


```text
@ARTICLE{2018arXiv180401438W,
    author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
    title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1804.01438},
    primaryClass = "cs.CV",
    keywords = {Computer Science - Computer Vision and Pattern Recognition},
    year = 2018,
    month = apr,
    adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
