# darknet_MGN #

> 2018.11.19

1、从DarkNet官方GitHub上copy过来源码


> 2018.11.22

2、增加了slice_layer，https://blog.csdn.net/lwplwf/article/details/84339881

3、修改了maxpool_layer的接口，支持了多尺度kernel size和stride size


> 2018.11.28

4、修改Python接口调用，支持numpy图片输入，参考demo_MGN.py，https://blog.csdn.net/lwplwf/article/details/84566954


> 2018.11.30

5、修改了maxpool_layer计算方式，和Caffe保持一致
