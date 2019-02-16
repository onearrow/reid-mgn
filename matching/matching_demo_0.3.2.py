#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: lwp
@last update: 2019/02/15
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
import matplotlib.pyplot as plt


def Savefile(img_name, fea, i, id):
    id_path = "%s/result" %DATA_SET+"/%d" %(id)
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
    # Candidate ID
    ID_candidate_list = []
    dist_candidate_list = []
    diff_list = []
    for j in range(len(distmat)):
        if distmat[j]<THRESHOLD:
            if ID_candidate_list.count(id_base[j]) == 0:
                ID_candidate_list.append(id_base[j])
                dist_candidate_list.append(distmat[j])
    # ID internal judgment
    for m in range(len(ID_candidate_list)):
        ID_candidate = ID_candidate_list[m]
        
        # ID internal feature list
        ID_feature_list = []
        ID_feature_index = [s for s, x in enumerate(id_base) if x == ID_candidate]
        for n in range(len(ID_feature_index)):
            feature_index = ID_feature_index[n]
            ID_feature = feature_base[feature_index]
            ID_feature_list.append(ID_feature)
                    
        ID_distmat = (cdist(feature, ID_feature_list))[0]
        # Exclude maximum and minimum
        if len(ID_distmat)>3:
            ID_distmat = ID_distmat.tolist()
            ID_distmat.remove(max(ID_distmat))
            ID_distmat.remove(min(ID_distmat))
        # Calculate average distance and judge
        average_dist = float(sum(ID_distmat))/len(ID_distmat)
        diff = abs(dist_candidate_list[m]-average_dist)
        diff_list.append(diff)
    if len(diff_list)>0:
        index = diff_list.index(min(diff_list))
        if min(diff_list)<THRESHOLD_ClUSTERING:
            NEW_ID = False
            ID = ID_candidate_list[index]
    
    return ID, NEW_ID


def Matching(feature_data_path, image_data_path):
    # Load data
    with open(feature_data_path,"r") as f1:
        feature_data = json.load(f1)
    with open(image_data_path,"r") as f2:
        image_data = json.load(f2)

    print "image num: %d" % (len(image_data))
    print "feature num: %d" % (len(feature_data))
    
    if os.path.exists("%s/result" %DATA_SET):
        shutil.rmtree("%s/result" %DATA_SET)

    # Matching
    id_base = []
    id_num  = 0
    t_list  = []
    for i in range(len(feature_data)):
        tic = timeit.default_timer()
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
                id_base.append(ID)
                feature_base = np.vstack((feature_base, feature))
                Savefile(img_name, fea, i, ID)
                print "i: %d, id num: %d" %(i, id_num)
            else:
                id_base.append(ID)
                feature_base = np.vstack((feature_base, feature))
                Savefile(img_name, fea, i, ID)
                print "i: %d, id num: %d, identical id: %d<<<%s" %(i, id_num, ID, img_name)
        toc = timeit.default_timer()
        t = int((toc-tic)*1000)
        print "time consuming: %d" %t
        t_list.append(t)

    if TIME_CONSUMING:
        plt.xlabel("feature index")
        plt.ylabel("time consuming(ms)")
        plt.plot(t_list)
        plt.savefig("test_time.jpg")

    print "id num: %d" %(id_num)    
    print "Matching completed!"
    
    return 0


def Test(TEST_PATH):
    id_list = os.listdir(TEST_PATH)
    class_acc_sum = 0
    false_id_sum  = 0
    for id_name in id_list:
        image_list = os.listdir(TEST_PATH+'/'+id_name)
        image_id_list = []
        for image_name in image_list:
            image_id = (image_name.split('_', 1))[0]
            image_id_list.append(image_id)
        ID_name = image_id_list[0]
        true_id_num  = float(image_id_list.count(ID_name))
        false_id_num = len(image_id_list)-true_id_num
        false_id_sum = false_id_sum+false_id_num
        class_acc  = float(true_id_num)/len(image_id_list)
        class_acc_sum = class_acc_sum+class_acc
    ID_acc = 1-float(abs(len(id_list)-GROUDTRUTH_ID_NUM))/GROUDTRUTH_ID_NUM # 1
    CLASS_acc = class_acc_sum/len(id_list) # 2
    FALSE_ID_error = false_id_sum # 3
    print "ID acc: %f" %ID_acc
    print "Class acc: %f" %CLASS_acc
    print "FALSE ID error: %d" %FALSE_ID_error


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
    THRESHOLD            = 0.87
    THRESHOLD_ClUSTERING = 0.4
    GROUDTRUTH_ID_NUM    = 212 # 750 212 7
    DATA_SET = "H"

    EXTRACT  = False
    MATCHING = True
    TEST     = True
     
    SAVE_FEATURE   = False
    COPY_IMAGE     = True
    TIME_CONSUMING = True

    IMAGE_PATH       = "%s/bounding_box_test" %DATA_SET
    MODEL_PATH       = "model/MGN_01_11_M_H.pt"
    FEATURE_MAT_PATH = "%s/feature_mat.json" %DATA_SET
    IMAGE_MAT_PATH   = "%s/image_mat.json" %DATA_SET
    TEST_PATH        = os.getcwd() + '/%s/result' %DATA_SET
    
    if EXTRACT:
        Generatematfile(MODEL_PATH, IMAGE_PATH)
    if MATCHING:
        tic = timeit.default_timer()
        Matching(FEATURE_MAT_PATH, IMAGE_MAT_PATH)
        toc = timeit.default_timer()
        print 'Matching time: %.2f' %(toc-tic)
    if TEST:
        Test(TEST_PATH)
