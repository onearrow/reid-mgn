# re-id_mgn
Re-ID MGN网络从Pytorch转换到Caffe，再转到DarkNet

在caffe2darknet的转换中，slice层由于DarkNet不支持，需要修改DarkNet源码添加新层实现，且需要修改源码让pooling层支持多尺度kernel size和stride size。（修改后的DarkNet源码稍后我会放出来）
