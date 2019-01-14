#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import average_precision_score


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]):
            continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1) # distmat:3368*15913,按行从小到大排序（即对每个id的特征距离排序）,返回排序后的索引,索引对应的实质上是gallery中图片的位置
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]) # gallery按特征距离排序后和query进行id匹配，得到真实label(还未清除同id同相机的情况)
    # Compute AP for each query
    aps = []
    
    # distence threshold
    t_list = [10, 7, 5, 3, 1]
    for k in range(len(t_list)):
        top = t_list[k]
        precision = []
        score = []

        # print "m, n:", m, n # 3368, 15913
        for i in range(m):
            # print "i:", i
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                     (gallery_cams[indices[i]] != query_cams[i]))
            
            y_true = matches[i, valid] # [ True  True  True ... False False False] 15913->15905,过滤后的真实label
            y_score = -distmat[i][indices[i]][valid] # [-0.60560741 -0.63570677 -0.63673882...-1.47429787 -1.48229123 -1.48770559] 15905
            
            if not np.any(y_true):
                continue
            if not aps:
                aps.append(average_precision_score(y_true, y_score)) # y_true:真实标签,y_score:预测标签

            # distence threshold
            t = y_true[0:top]
            s = y_score[0:top]
            precision.append(np.sum(t==True)/len(t))
            score.append(s[-1])
        
        # distence threshold
        # print "top-", top
        # print "precision:", np.mean(precision)
        # print "distence threshold:", np.mean(score)
        # print "---------------------------"

    if len(aps) == 0:
        raise RuntimeError("No valid query")

    return np.mean(aps)
