#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from convert import mgn_res


if __name__=='__main__':
    # np.set_printoptions(precision=4, threshold='nan')

    # load model
    net = mgn_res.MGN()
    # print(net)

    checkpoint = torch.load('model_best.pt')
    net.load_state_dict(checkpoint)
    net.eval()

    img = Image.open("test.jpg") # RGB
    print(img.size)
    img = img.resize((128, 384))
    print(img.size)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    img = transform(img)
    img = img.unsqueeze_(0).float()
    # print(img)

    input_var = Variable(img)
    
    output_var = net(input_var)
    print(output_var.data.numpy(), len(output_var[0]))
    
    print("Done!")
