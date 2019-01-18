#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: lwp
@last update: 2019/01/18
@function: 
'''
import os
import cv2
import json
import timeit
import numpy as np
import shutil
import torch
from scipy.spatial.distance import cdist
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import mgn


def Savefile(img_name, fea, i, id):
    id_path = "result"+"/%d" %(id)
    feature_name = "%d.json" %(i)

    if not os.path.exists(id_path): 
        os.makedirs(id_path)

    if SAVE_FEATURE:
        with open(id_path+"/"+feature_name, "w") as f:
            json.dump(fea, f)
    if COPY_IMAGE:
        shutil.copyfile(IMAGE_PATH+"/"+img_name, id_path+"/"+img_name)
    
    return 0


def Clustering(i, feature, id_base, feature_base):
    NEW_ID = True
    ID = i
    distmat = (cdist(feature, feature_base))[0]
    # print distmat
    for j in range(len(distmat)):
        if distmat[j]<THRESHOLD:
            ID = id_base[j]
            ID_index = [s for s, x in enumerate(id_base) if x == ID]
                    
            ID_feature_list = []
            for k in range(len(ID_index)):
                ID_feature_index = ID_index[k]
                ID_feature = feature_base[ID_feature_index]
                ID_feature_list.append(ID_feature)
                    
            ID_distmat   = (cdist(feature, ID_feature_list))[0]
            average_dist = float(sum(ID_distmat))/len(ID_distmat)
                    
            if abs(distmat[j]-average_dist)<THRESHOLD_ClUSTERING:
                NEW_ID = False
                break
    
    return ID, NEW_ID


def Matching(feature_data_path, image_data_path):
    # load data
    with open("feature_mat1.json","r") as f1:
        feature_data = json.load(f1)
    with open("image_mat1.json","r") as f2:
        image_data = json.load(f2)

    print "image num: %d" % (len(image_data))
    print "feature num: %d" % (len(feature_data))
    
    if os.path.exists("result"):
        shutil.rmtree("result")

    # matching
    id_base = []
    id_num  = 0
    for i in range(len(feature_data)):
        img_name = image_data[i]
        fea      = feature_data[i]
        feature  = np.array(fea)
        feature  = np.reshape(feature, (1, 2048))
        if i==0:
            ID = i
            id_num += 1
            id_base.append(ID)
            feature_base = feature
            Savefile(img_name, fea, i, ID)
        else:
            ID, NEW_ID = Clustering(i, feature, id_base, feature_base)
            if NEW_ID:
                id_num += 1
                id_base.append(I                        D)
                feature_base = np.vstack((feature_base, feature))
                Savefile(img_name, fea, i, ID)
                print "i: %d, id num: %d" %(i, id_num)
            else:
                id_base.append(ID)
                feature_base = np.vstack((feature_base, feature))
                Savefile(img_name, fea, i, ID)
                print "i: %d, id num: %d, identical id: %d<<<%s" %(i, id_num, ID, img_name)
    print "id num: %d" %(id_num)    
    print "Matching completed!"
    
    return 0


def Generatematfile(MODEL_PATH, IMAGE_PATH):
    extractor  = pfextractor(MODEL_PATH)
    image_list = os.listdir(IMAGE_PATH)
    
    with open(IMAGE_MAT_PATH, "w") as f1:
        json.dump(image_list, f1)

    i = 0
    features = []
    for image_name in image_list:
        image_path = IMAGE_PATH+'/'+image_name
        image = cv2.imread(image_path)
        feature = extractor.extract(image)
        features.append(feature)
        i += 1
        print "i: %d, feature num: %d, %s id done!" %(i, len(features), image_name)
    
    with open(FEATURE_MAT_PATH, "w") as f2:
        json.dump(features, f2)
    
    print "Generating matrix file completed!"
    
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
    THRESHOLD = 0.91
    THRESHOLD_ClUSTERING = 0.5

    EXTRACT      = False
    SAVE_FEATURE = False
    COPY_IMAGE   = True

    IMAGE_PATH       = "bounding_box_test"
    MODEL_PATH       = "model/MGN_01_11_M_H.pt"
    FEATURE_MAT_PATH = "feature_mat.json"
    IMAGE_MAT_PATH   = "image_mat.json"
    
    if EXTRACT:
        Generatematfile(MODEL_PATH, IMAGE_PATH)
    else:
        tic = timeit.default_timer()
        Matching(FEATURE_MAT_PATH, IMAGE_MAT_PATH)
        toc = timeit.default_timer()
        print 'Matching time: %.2f' %(toc-tic)
