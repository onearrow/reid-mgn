# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck,resnet50
import torch.nn as nn


class MGN(nn.Module):
    def __init__(self, num_classes = 751):
        super(MGN, self).__init__()

        # 使用resnet的的前面层作为基础特征特征提取结构,分支结构共享部分
        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,    # res_conv2
            resnet.layer2,    # res_conv3
            resnet.layer3[0], # res_conv4_1
        )

        # MGN Network,The difference is that we employ no down-sampling operations in res_conv5_1 block.
        
        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        # res_conv5 global
        res_g_conv5 = resnet.layer4
        # res_conv5 part
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        # mgn part-1 global
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        # mgn part-2
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        # mgn part-3
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # global max pooling
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2   = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3   = nn.MaxPool2d(kernel_size=(8, 8))

        # 每个分支中用于降维的1×1conv和用于identity prediction的全连接层不共享权重

        # conv1 reduce
        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # fc softmax loss
        self.fc_id_2048_0 = nn.Linear(256, num_classes) # 2048
        self.fc_id_2048_1 = nn.Linear(256, num_classes) # 2048
        self.fc_id_2048_2 = nn.Linear(256, num_classes) # 2048
        self.fc_id_256_1_0 = nn.Linear(256, num_classes)
        self.fc_id_256_1_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_0 = nn.Linear(256, num_classes)
        self.fc_id_256_2_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_2 = nn.Linear(256, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)
        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.constant_(fc.bias, 0.)


    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: predict
        """

        # 基础模型特征提取部分,resnet50 conv4_2之前的结构
        x = self.backone(x)

        # 构建三个分支结构并进行第一次特征pool,分支1特征图缩小,分支2分支3特征图大小不变
        # p1_single
        p1 = self.p1(x)
        zg_p1 = self.maxpool_zg_p1(p1)
        
        # p2_single
        p2 = self.p2(x)
        zg_p2 = self.maxpool_zg_p2(p2)

        # p3_single
        p3 = self.p3(x)
        zg_p3 = self.maxpool_zg_p3(p3)

        # 对分支2处理,pool得到1个feature_map,特征分割得到2个feature_map,共3个feature_map
        zp2 = self.maxpool_zp2(p2)
        z0_p2,z1_p2 = torch.split(zp2, 1 ,2)
        
        # 对分支3处理,pool得到1个feature_map,特征分割得到3个feature_map,共4个feature_map
        zp3 = self.maxpool_zp3(p3)
        z0_p3,z1_p3,z2_p3 = torch.split(zp3,1,2)    

        # 分支1,分支2,分支3获取到的8个feature_map
        # p1_single
        fg_p1 = self.reduction_0(zg_p1)
        
        # p2_single
        fg_p2 = self.reduction_1(zg_p2)
        
        f0_p2 = self.reduction_3(z0_p2)
        f1_p2 = self.reduction_4(z1_p2)

        # p3_single
        fg_p3 = self.reduction_2(zg_p3)
        
        f0_p3 = self.reduction_5(z0_p3)
        f1_p3 = self.reduction_6(z1_p3)
        f2_p3 = self.reduction_7(z2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        
        return predict
