# Re-ID MGN模型

## 1. 模型转换

### --[pytorch2caffe_MGN](https://github.com/lwplw/re-id_mgn/tree/master/pytorch2caffe_MGN)
- [x] 结构文件转换
- [x] 权重文件转换
- [x] 预测结果验证

### --[caffe2darknet_MGN](https://github.com/lwplw/re-id_mgn/tree/master/caffe2darknet_MGN)
- [x] 结构文件转换
- [x] 权重文件转换
- [x] 预测结果验证

### --[darknet_MGN](https://github.com/lwplw/re-id_mgn/tree/master/darknet_MGN)

修改源码之后的DarkNet


## 2. 模型训练

### --[pytorch_MGN](https://github.com/lwplw/re-id_mgn/tree/master/pytorch_MGN)

MGN模型的训练，在Pytorch下进行


## Remark

(1)在caffe2darknet的转换中，slice层由于DarkNet不支持，需要修改DarkNet源码添加新层实现

(2)需要修改源码让max_pool层支持多尺度kernel size和stride size

(3)修改DarkNet源码，增加了Python借口调用对numpy格式输入的支持

