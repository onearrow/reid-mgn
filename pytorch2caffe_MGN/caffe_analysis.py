# coding=utf-8
import sys
caffe_root = '/home/lwp/beednprojects/caffe-2.7/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np


if __name__=="__main__":

    np.set_printoptions(threshold='nan')
    
    network = "MGN.prototxt"
    model   = "MGN.caffemodel"
    output1  = "MGN_caffe_analysis_blobs.txt"
    output2  = "MGN_caffe_analysis_params.txt"
    pf1 = open(output1, 'w')
    pf2 = open(output2, 'w')

    net = caffe.Net(network, model, caffe.TEST)

    # blobs
    for param_name, blob in net.blobs.iteritems():
    	 pf1.write(param_name + '\t' + str(blob.data.shape) + '\n')

    # params
    for param_name, param in net.params.iteritems():
        if len(net.params[param_name]) > 1:
            pf2.write(param_name + '\t' + "w:" + str(param[0].data.shape) + '\t' + "b:" + str(param[1].data.shape) + '\n')
    	else:
            pf2.write(param_name + '\t' + "w:" + str(param[0].data.shape) + '\t' + "b:()" + '\n')
    	
    pf1.close
    pf2.close
    print("Done!")
