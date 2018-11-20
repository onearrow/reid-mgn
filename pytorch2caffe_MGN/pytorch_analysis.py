import sys
sys.path.insert(0,'.')
import torch
import torch.nn as nn
from torch.autograd import Variable
from convert import mgn_res
from torchsummary import summary

if __name__=='__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = mgn_res.MGN().to(device)
    print(net)
    summary(net, (3, 384, 128))

    