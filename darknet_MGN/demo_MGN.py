#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
# import pdb

from ctypes import *
import math
import random

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

free_image = lib.free_image
free_image.argtypes = [IMAGE]


def MGN(net, image):
    im = load_image(image, 0, 0)
    out = predict_image(net, im)

    res = []
    for i in range(2048):
        res.append(out[i])
        print(i, out[i])
    
    free_image(im)
    return res

if __name__ == '__main__':
    net = load_net("cfg/MGN.cfg", "backup/MGN.weights", 0)
    test_img = "data/test.jpg"
    
    res = MGN(net, test_img)
    print(len(res))
    print("Done!")
    