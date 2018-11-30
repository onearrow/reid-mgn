#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from cfg import *
from prototxt import *
import numpy as np

def caffe2darknet(protofile, caffemodel):
    model = parse_caffemodel(caffemodel)
    layers = model.layer
    if len(layers) == 0:
        print 'Using V1LayerParameter'
        layers = model.layers
    
    lmap = {}
    for l in layers:
       lmap[l.name] = l 

    net_info = parse_prototxt(protofile)
    props = net_info['props']

    wdata = []
    blocks = []
    block = OrderedDict()
    block['type'] = 'net'
    if props.has_key('input_shape'):
        block['batch'] = props['input_shape']['dim'][0]
        block['channels'] = props['input_shape']['dim'][1]
        block['height'] = props['input_shape']['dim'][2]
        block['width'] = props['input_shape']['dim'][3]
    else:
        block['batch'] = props['input_dim'][0]
        block['channels'] = props['input_dim'][1]
        block['height'] = props['input_dim'][2]
        block['width'] = props['input_dim'][3]
    if props.has_key('mean_file'):
        block['mean_file'] = props['mean_file']
    blocks.append(block)

    layers = net_info['layers']
    layer_num = len(layers)
    i = 0 # layer id
    layer_id = dict()
    layer_id[props['input']] = 0
    while i < layer_num:
        layer = layers[i]
        print i,layer['name'], layer['type']
        if layer['type'] == 'Convolution':
            if layer_id[layer['bottom']] != len(blocks)-1: # 不是接上一层
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                print("route", layer_id[layer['bottom']], len(blocks), block['layers'])
                blocks.append(block)
            conv_layer = layers[i]
            block = OrderedDict()
            block['type'] = 'convolutional'
            block['filters'] = conv_layer['convolution_param']['num_output']
            block['size'] = conv_layer['convolution_param']['kernel_size']
            block['stride'] = conv_layer['convolution_param']['stride']
            block['pad'] = conv_layer['convolution_param']['pad']
            last_layer = conv_layer 
            m_conv_layer = lmap[conv_layer['name']] 
            if i+2 < layer_num and layers[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print i+1, layers[i+1]['name'], layers[i+1]['type']
                print i+2, layers[i+2]['name'], layers[i+2]['type']
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                m_scale_layer = lmap[scale_layer['name']]
                m_bn_layer = lmap[bn_layer['name']]
                wdata += list(m_scale_layer.blobs[1].data)  ## conv_bias <- sc_beta
                wdata += list(m_scale_layer.blobs[0].data)  ## bn_scale  <- sc_alpha
                wdata += (np.array(m_bn_layer.blobs[0].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_mean <- bn_mean/bn_scale
                wdata += (np.array(m_bn_layer.blobs[1].data) / m_bn_layer.blobs[2].data[0]).tolist()  ## bn_var  <- bn_var/bn_scale
                i = i + 2
            else:
                wdata += list(m_conv_layer.blobs[1].data)   ## conv_bias
            wdata += list(m_conv_layer.blobs[0].data)       ## conv_weights
            
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                print i+1, layers[i+1]['name'], layers[i+1]['type']
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = last_layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

        elif layer['type'] == 'Pooling':
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                print("route", layer_id[layer['bottom']], len(blocks), block['layers'])
                blocks.append(block)
            block = OrderedDict()
            if layer['pooling_param']['pool'] == 'AVE':
                block['type'] = 'avgpool'
            elif layer['pooling_param']['pool'] == 'MAX':
                block['type'] = 'maxpool'                
            if layer['pooling_param'].has_key('kernel_size'):
                block['size_h'] = layer['pooling_param']['kernel_size']
                block['size_w'] = layer['pooling_param']['kernel_size']
            else:
                block['size_h'] = layer['pooling_param']['kernel_h']
                block['size_w'] = layer['pooling_param']['kernel_w']
            if layer['pooling_param'].has_key('stride'):
                block['stride_h'] = layer['pooling_param']['stride']
                block['stride_w'] = layer['pooling_param']['stride']
            else:
                block['stride_h'] = layer['pooling_param']['stride_h']
                block['stride_w'] = layer['pooling_param']['stride_w']
            if layer['pooling_param'].has_key('pad'):
                block['padding'] = layer['pooling_param']['pad']
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

        elif layer['type'] == 'Eltwise':
            bottoms = layer['bottom']
            bottom1 = layer_id[bottoms[0]] - len(blocks)
            bottom2 = layer_id[bottoms[1]] - len(blocks)
            assert(bottom1 == -1 or bottom2 == -1)
            from_id = bottom2 if bottom1 == -1 else bottom1
            block = OrderedDict()
            block['type'] = 'shortcut'
            block['from'] = str(from_id)
            assert(i+1 < layer_num and layers[i+1]['type'] == 'ReLU')
            print i+1, layers[i+1]['name'], layers[i+1]['type']
            block['activation'] = 'relu'
            top = layers[i+1]['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 2

        # elif layer['type'] == 'Slice':
        #     assert(layer_id[layer['bottom']] == len(blocks)-1)
        #     block = OrderedDict()
        #     block['type'] = 'slice'
        #     block['axis'] = layer['slice_param']['axis']
        #     tops = layer['top']
        #     if len(tops) == 2:
        #         layer_id[tops[0]] = len(blocks)
        #         layer_id[tops[1]] = len(blocks) + 1
        #     if len(tops) == 3:
        #         layer_id[tops[0]] = len(blocks)
        #         layer_id[tops[1]] = len(blocks) + 1
        #         layer_id[tops[2]] = len(blocks) + 2
        #     blocks.append(block)
        #     i = i + 1

        elif layer['type'] == 'Slice':
            tops = layer['top']
            for n in range(len(tops)):
                # 不是接上一层
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                print("route", layer_id[layer['bottom']], len(blocks), block['layers'])
                blocks.append(block)

                block = OrderedDict()
                block['type'] = 'slice'
                block['slice_axis'] = layer['slice_param']['axis']
                block['slice_num'] = len(tops) - 1
                block['slice_pos'] = n
                layer_id[tops[n]] = len(blocks)
                blocks.append(block)
            i = i + 1

        elif layer['type'] == 'Concat':
            bottoms = layer['bottom']
            bottom1 = str(layer_id[bottoms[0]] - len(blocks))
            bottom2 = str(layer_id[bottoms[1]] - len(blocks))
            bottom3 = str(layer_id[bottoms[2]] - len(blocks))
            bottom4 = str(layer_id[bottoms[3]] - len(blocks))
            bottom5 = str(layer_id[bottoms[4]] - len(blocks))
            bottom6 = str(layer_id[bottoms[5]] - len(blocks))
            bottom7 = str(layer_id[bottoms[6]] - len(blocks))
            bottom8 = str(layer_id[bottoms[7]] - len(blocks))
            block = OrderedDict()
            block['type'] = 'route'
            block['layers'] = bottom1+','+bottom2+','+bottom3+','+bottom4+','+bottom5+','+bottom6+','+bottom7+','+bottom8
            # print(block['layers']) 
            blocks.append(block)
            i = i + 1

        elif layer['type'] == 'InnerProduct':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'connected'
            block['output'] = layer['inner_product_param']['num_output']
            m_fc_layer = lmap[layer['name']]
            wdata += list(m_fc_layer.blobs[1].data) # fc_bias
            wdata += list(m_fc_layer.blobs[0].data) # fc_weights
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

        elif layer['type'] == 'Softmax':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'softmax'
            block['groups'] = 1
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

        else:
            print('unknown type %s' % layer['type'])
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            block = OrderedDict()
            block['type'] = layer['type']
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

    print 'done' 
    return blocks, np.array(wdata)

def prototxt2cfg(protofile):
    net_info = parse_prototxt(protofile)
    props = net_info['props']

    blocks = []
    block = OrderedDict()
    block['type'] = 'net' 
    if props.has_key('input_shape'): 
        block['batch'] = props['input_shape']['dim'][0]
        block['channels'] = props['input_shape']['dim'][1]
        block['height'] = props['input_shape']['dim'][2]
        block['width'] = props['input_shape']['dim'][3]
    else:
        block['batch'] = props['input_dim'][0]
        block['channels'] = props['input_dim'][1]
        block['height'] = props['input_dim'][2]
        block['width'] = props['input_dim'][3]
    if props.has_key('mean_file'):
        block['mean_file'] = props['mean_file']
    blocks.append(block)

    layers = net_info['layers']
    layer_num = len(layers)
    i = 0 # layer id
    layer_id = dict()
    layer_id[props['input']] = 0
    while i < layer_num:
        layer = layers[i]
        print i,layer['name'], layer['type']
        if layer['type'] == 'Convolution':
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            conv_layer = layers[i]
            block = OrderedDict()
            block['type'] = 'convolutional'
            block['filters'] = conv_layer['convolution_param']['num_output']
            block['size'] = conv_layer['convolution_param']['kernel_size']
            block['stride'] = '1'
            if conv_layer['convolution_param'].has_key('stride'):
                block['stride'] = conv_layer['convolution_param']['stride']
            block['pad'] = '1'
            last_layer = conv_layer 
            if i+2 < layer_num and layers[i+1]['type'] == 'BatchNorm' and layers[i+2]['type'] == 'Scale':
                print i+1,layers[i+1]['name'], layers[i+1]['type']
                print i+2,layers[i+2]['name'], layers[i+2]['type']
                block['batch_normalize'] = '1'
                bn_layer = layers[i+1]
                scale_layer = layers[i+2]
                last_layer = scale_layer
                i = i + 2
            
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                print i+1,layers[i+1]['name'], layers[i+1]['type']
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1
            else:
                block['activation'] = 'linear'
                top = last_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
            i = i + 1

        elif layer['type'] == 'Pooling':
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            block = OrderedDict()
            if layer['pooling_param']['pool'] == 'AVE':
                block['type'] = 'avgpool'
            elif layer['pooling_param']['pool'] == 'MAX':
                block['type'] = 'maxpool'                
            if layer['pooling_param'].has_key('kernel_size'):
                block['size'] = layer['pooling_param']['kernel_size']
            else:
                block['size_h'] = layer['pooling_param']['kernel_h']
                block['size_w'] = layer['pooling_param']['kernel_w']
            if layer['pooling_param'].has_key('stride'):
                block['stride'] = layer['pooling_param']['stride']
            else:
                block['stride_h'] = layer['pooling_param']['stride_h']
                block['stride_w'] = layer['pooling_param']['stride_w']
            if layer['pooling_param'].has_key('pad'):
                pad = int(layer['pooling_param']['pad'])
                if pad > 0:
                    block['pad'] = '1'
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

        elif layer['type'] == 'Eltwise':
            bottoms = layer['bottom']
            bottom1 = layer_id[bottoms[0]] - len(blocks)
            bottom2 = layer_id[bottoms[1]] - len(blocks)
            assert(bottom1 == -1 or bottom2 == -1)
            from_id = bottom2 if bottom1 == -1 else bottom1
            block = OrderedDict()
            block['type'] = 'shortcut'
            block['from'] = str(from_id)
            assert(i+1 < layer_num and layers[i+1]['type'] == 'ReLU')
            block['activation'] = 'relu'
            top = layers[i+1]['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 2

        # elif layer['type'] == 'Slice':
        #     assert(layer_id[layer['bottom']] == len(blocks)-1)
        #     block = OrderedDict()
        #     block['type'] = 'slice'
        #     tops = layer['top']
        #     if len(tops) == 2:
        #         layer_id[tops[0]] = len(blocks)
        #         layer_id[tops[1]] = len(blocks) + 1
        #     if len(tops) == 3:
        #         layer_id[tops[0]] = len(blocks)
        #         layer_id[tops[1]] = len(blocks) + 1
        #         layer_id[tops[2]] = len(blocks) + 2
        #     blocks.append(block)
        #     i = i + 1

        elif layer['type'] == 'Slice':
            tops = layer['top']
            for n in range(len(tops)):
                # 不是接上一层
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)

                block = OrderedDict()
                block['type'] = 'slice'
                block['axis'] = layer['slice_param']['axis']
                layer_id[tops[n]] = len(blocks)
                blocks.append(block)
            i = i + 1

        elif layer['type'] == 'Concat':
            bottoms = layer['bottom']
            bottom1 = str(layer_id[bottoms[0]] - len(blocks))
            bottom2 = str(layer_id[bottoms[1]] - len(blocks))
            bottom3 = str(layer_id[bottoms[2]] - len(blocks))
            bottom4 = str(layer_id[bottoms[3]] - len(blocks))
            bottom5 = str(layer_id[bottoms[4]] - len(blocks))
            bottom6 = str(layer_id[bottoms[5]] - len(blocks))
            bottom7 = str(layer_id[bottoms[6]] - len(blocks))
            bottom8 = str(layer_id[bottoms[7]] - len(blocks))
            block = OrderedDict()
            block['type'] = 'route'
            block['layers'] = bottom1+','+bottom2+','+bottom3+','+bottom4+','+bottom5+','+bottom6+','+bottom7+','+bottom8
            # print(block['layers']) 
            blocks.append(block)
            i = i + 1

        elif layer['type'] == 'InnerProduct':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'connected'
            block['output'] = layer['inner_product_param']['num_output']
            if i+1 < layer_num and layers[i+1]['type'] == 'ReLU':
                act_layer = layers[i+1]
                block['activation'] = 'relu'
                top = act_layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 2
            else:
                block['activation'] = 'linear'
                top = layer['top']
                layer_id[top] = len(blocks)
                blocks.append(block)
                i = i + 1

        elif layer['type'] == 'Softmax':
            assert(layer_id[layer['bottom']] == len(blocks)-1)
            block = OrderedDict()
            block['type'] = 'softmax'
            block['groups'] = 1
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

        else:
            print('unknown type %s' % layer['type'])
            if layer_id[layer['bottom']] != len(blocks)-1:
                block = OrderedDict()
                block['type'] = 'route'
                block['layers'] = str(layer_id[layer['bottom']] - len(blocks))
                blocks.append(block)
            block = OrderedDict()
            block['type'] = layer['type']
            top = layer['top']
            layer_id[top] = len(blocks)
            blocks.append(block)
            i = i + 1

    print 'done' 
    return blocks


def save_weights(data, weightfile):
    print 'Save to ', weightfile
    wsize = data.size
    weights = np.zeros((wsize+4,), dtype=np.int32)
    ## write info 
    weights[0] = 0
    weights[1] = 1
    weights[2] = 0      ## revision
    weights[3] = 0      ## net.seen
    weights.tofile(weightfile)
    weights = np.fromfile(weightfile, dtype=np.float32)
    # weights[4:] = data
    for i in range(wsize):
        weights[i+4] = data[i]
    weights.tofile(weightfile)

    # lwp-2018.11.29
    # print(len(data))
    # print(data[0])
    # print(data[wsize-1])

    # print(weights[0])
    # print(weights[1])
    # print(weights[2])
    # print(weights[3])
    # print(len(weights))

    # pf1 = open("okkkkk.txt", 'w')
    # for i in range(wsize):
    #     pf1.write('%f ' % data[i])
    #     if (i+1)%147 == 0:
    #         pf1.write('\n\n')
    # pf1.close
    
if __name__ == '__main__':
    import sys
    
    # if len(sys.argv) != 5:
    #     print('try:')
    #     print('python caffe2darknet.py MGN.prototxt MGN.caffemodel MGN.cfg MGN.weights')
    #     exit()
    # protofile  = sys.argv[1]
    # caffemodel = sys.argv[2]
    # cfgfile    = sys.argv[3]
    # weightfile = sys.argv[4]

    protofile  = "MGN.prototxt" 
    caffemodel = "MGN.caffemodel"
    cfgfile    = "MGN.cfg"
    weightfile = "MGN.weights"
    
    blocks, data = caffe2darknet(protofile, caffemodel)
    
    save_weights(data, weightfile)
    save_cfg(blocks, cfgfile)
    
    print_cfg(blocks)
    print_cfg_nicely(blocks)
