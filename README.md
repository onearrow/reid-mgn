# Re-ID MGN模型

## 1. 模型转换

### --pytorch2caffe_MGN

模型结构转换 -- 已完成
模型权重转换 -- 已完成
结果验证 -- 已完成

### --caffe2darknet_MGN

模型结果转换 -- 已完成
模型权重转换 -- 已完成
结果验证 -- 已完成

### --darknet_MGN

修改源码之后的DarkNet


## 2. 模型训练

### MGN_pytorch

MGN模型的训练，在Pytorch下进行


## remark

(1)在caffe2darknet的转换中，slice层由于DarkNet不支持，需要修改DarkNet源码添加新层实现

(2)需要修改源码让max_pool层支持多尺度kernel size和stride size

(3)修改DarkNet源码，增加了Python借口调用对numpy格式输入的支持

