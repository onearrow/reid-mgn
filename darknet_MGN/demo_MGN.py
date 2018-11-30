#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))
# import pdb

from ctypes import *
import math
import random
import numpy as np
import cv2
from PIL import Image


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

def nparray_to_image(img): 
    data = img.ctypes.data_as(POINTER(c_ubyte)) 
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image

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

ndarray_image = lib.ndarray_to_image 
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)] 
ndarray_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

free_image = lib.free_image
free_image.argtypes = [IMAGE]


def MGN(net, image):
    # PIL Image
    image = Image.open(image_path) # RGB
    image = image.resize((128, 384))
    # image.show()
    image = np.array(image)
    r = cv2.split(image)[0]
    g = cv2.split(image)[1]
    b = cv2.split(image)[2]
    image = cv2.merge([b, g, r])
    # image = image / 255.0 # nparray_to_image内部会处理
    im = nparray_to_image(image) #  由numpy转换得到IMAGE

    # IMAGE
    # im = load_image(image_path, 0, 0) # 由路径加载得到IMAGE
    out = predict_image(net, im)
    
    res = []
    for i in range(2048):
        res.append(out[i])
    print(res)
    
    free_image(im)
    return res

if __name__ == '__main__':
    net = load_net("cfg/MGN.cfg", "backup/MGN.weights", 0)
    test_img = "data/test.jpg"
    
    res = MGN(net, test_img)
    print(len(res))
    print("Done!")
    
