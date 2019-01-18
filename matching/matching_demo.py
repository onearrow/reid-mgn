#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: lwp
@last update: 2019/01/18
@function: 
'''
import os
import json
import numpy as np
import shutil
import cv2
import torch
from scipy.spatial.distance import cdist
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import mgn
import timeit


def Savefeature(fimg, img_name, fea, i, id):
    feature_path = "result"+"/%d" %(id)
    feature_name = "%d.json" %(i)
    if not os.path.exists(feature_path): 
        os.makedirs(feature_path)
    with open(feature_path+"/"+feature_name, "w") as f:
        json.dump(fea, f)
    print >> fimg, "%d, %s" %(id, img_name)

def Matching(feature_data, image_data, threshold):
    print "image num: %d" % (len(image_data))
    print "image num: %d" % (len(feature_data))

    if os.path.exists("result"):
        shutil.rmtree("result")
    if os.path.isfile("id_image.txt"):
        os.remove("id_image.txt")
    
    fimg = open("id_image.txt", "w")

    id_base = []
    id_num  = 0
    for i in range(len(feature_data)):
        ID       = i
        NEW_ID   = True
        img_name = image_data[i]
        fea      = feature_data[i]
        feature  = np.array(fea)
        feature  = np.reshape(feature, (1, 2048))
        if i==0:
            id_num += 1
            id_base.append(ID)
            feature_base = feature
            Savefeature(fimg, img_name, fea, i, ID)
        else:
            distmat = (cdist(feature, feature_base))[0]
            # exit(0)
            for j in range(len(distmat)):
                if distmat[j]<threshold:
                    NEW_ID = False
                    ID = id_base[j]
                    break
            if NEW_ID:
                id_num += 1
                id_base.append(ID)
                feature_base = np.vstack((feature_base, feature))
                Savefeature(fimg, img_name, fea, i, ID)
                print "i: %d, id num: %d" %(i, id_num)
            else:
                id_base.append(ID)
                feature_base = np.vstack((feature_base, feature))
                Savefeature(fimg, img_name, fea, i, ID)
                print "i: %d, id num: %d, identical id: %d<<<%s" %(i, id_num, ID, img_name)
    
    print "id num: %d" %(id_num)
    
    fimg.close()
    
    return 0


class pfextractor():
    def __init__(self, model_path):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU index
        self.model = mgn.MGN().cuda()
        self.model.load_state_dict(torch.load(model_path))
        
        self.transform = transforms.Compose([
                transforms.Resize((384, 128), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def extract(self, image):
        self.model.eval()
        
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        image = self.transform(image)
        image = image.unsqueeze_(0).float()
        image = Variable(image)

        output = self.model(image.cuda())
        
        f = output[0].data.cpu()
        fnorm = torch.norm(f)
        f = f.div(fnorm.expand_as(f))
                
        return f.tolist()
        

if __name__ == "__main__":

    EXTRACT = False
    if EXTRACT:
        path = "test_gallery/bounding_box_test"
        extractor = pfextractor("model/MGN_12_27.pt")

        image_list = os.listdir(path)
        with open("image_mat.json", "w") as f1:
            json.dump(image_list, f1)

        i = 0
        features = []
        for image_name in image_list:
            image_path = path+'/'+image_name
            image = cv2.imread(image_path)
            feature = extractor.extract(image)
            features.append(feature)
            i += 1
            print "i: %d, feature num: %d, %s id done!" %(i, len(features), image_name)
        with open("feature_mat.json", "w") as f2:
            json.dump(features, f2)
    else:
        threshold = 0.16

        with open("feature_mat.json","r") as f1:
            feature_data = json.load(f1)
        with open("image_mat.json","r") as f2:
            image_data = json.load(f2)

        Counting(feature_data, image_data, threshold)
