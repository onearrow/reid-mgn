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

#### （1）In the caffe2darknet conversion, need to modify the DarkNet source code to add the slice layer.
#### （2）The source code needs to be modified to allow the max_pool layer to support multi-scale kernel size and stride size
#### （3）Modify DarkNet source code to add support for numpy format for picture input in Python interface.

## 2. Model training
### -[pytorch_MGN](https://github.com/lwplw/re-id_mgn/tree/master/pytorch_MGN)
#### Train the MGN model under Pytorch.

## 3. Re-id matching module
### -[matching](https://github.com/lwplw/re-id_mgn/tree/master/matching)

## 4. Re-id attention MGN
### -[attention_MGN](https://github.com/lwplw/re-id_mgn/tree/master/attention_MGN)
#### Attempt to add attention module to the network structure of MGN.


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
