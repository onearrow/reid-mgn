# re-id_mgn
Re-ID MGN模型从Pytorch转换到DarkNet分两步进行：

（1）Pytorch->Caffe（模型结构转换、权重转换、验证）

（2）Caffe->DarkNet（模型结构转换、权重转换、验证（正在进行））

在caffe2darknet的转换中，slice层由于DarkNet不支持，需要修改DarkNet源码添加新层实现，且需要修改源码让pooling层支持多尺度kernel size和stride size。（修改后的DarkNet源码稍后我会放出来）
