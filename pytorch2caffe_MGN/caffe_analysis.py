# coding=utf-8
import sys
caffe_root = '/home/lwp/beednprojects/caffe-2.7/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np


if __name__=="__main__":

    np.set_printoptions(threshold='nan')
    
    BLOBS = False
    PARAMS = False
    WEIGHTS = True

    network = "MGN.prototxt"
    model   = "MGN.caffemodel"

    net = caffe.Net(network, model, caffe.TEST)


    # blobs
    if BLOBS:
        output1 = "MGN_caffe_analysis_blobs.txt"
        pf1 = open(output1, 'w')

        for param_name, blob in net.blobs.iteritems():
        	 pf1.write(param_name + '\t' + str(blob.data.shape) + '\n')
        
        pf1.close
        print("blobs is ok!")

    # params
    if PARAMS:
        output2 = "MGN_caffe_analysis_params.txt"
        pf2 = open(output2, 'w')

        for param_name, param in net.params.iteritems():
            if len(net.params[param_name]) > 1:
                pf2.write(param_name + '\t' + "w:" + str(param[0].data.shape) + '\t' + "b:" + str(param[1].data.shape) + '\n')
            else:
                pf2.write(param_name + '\t' + "w:" + str(param[0].data.shape) + '\t' + "b:()" + '\n')

        pf2.close
    	print("params is ok!")

    # weights
    if WEIGHTS:
        output3 = "MGN_caffe_analysis_weights.txt"
        pf3 = open(output3, 'w')

        # 遍历每一层
        for param_name in net.params.keys():
            weight = net.params[param_name][0].data
            
            # print(len(net.params[param_name][0].data[0][0][0]))
            # exit(0)
 
            # 该层在prototxt文件中对应“top”的名称
            pf3.write(param_name)
            pf3.write('\n')
 
            # 写权重参数
            pf3.write('\n' + param_name + '_weight:\n\n')
            # 权重参数是多维数组，为了方便输出，转为单列数组
            weight.shape = (-1, 1)
            
            # 7*7*64=3136
            cc = 0
            for n in range(64):
                for c in range(3):
                    for size_h in range(7):
                        for size_w in range(7):
                            pf3.write('%f ' % net.params[param_name][0].data[n][c][size_h][size_w])
                            cc = cc + 1
                pf3.write('\n\n')
            print(cc)
            # for w in weight:
            #     pf3.write('%f, ' % w)


            if len(net.params[param_name]) >1:
                bias = net.params[param_name][1].data
                # 写偏置参数
                pf3.write('\n\n' + param_name + '_bias:\n\n')
                # 偏置参数是多维数组，为了方便输出，转为单列数组
                bias.shape = (-1, 1)
                
                for b in bias:
                    pf3.write('%f, ' % b)
 
            pf3.write('\n\n')
 
        pf3.close
        print("weights is ok!")
    
    print("Done!")
