import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from convert import pytorch_to_caffe
from convert import mgn_res


if __name__=='__main__':
    name="MGN"

    # load model
    net = mgn_res.MGN()
    print(net)

    checkpoint = torch.load('model_best.pt')
    net.load_state_dict(checkpoint)
    net.eval()

    input_var = Variable(torch.rand([1, 3, 384, 128]))
    # output_var = net.forward(input_var)

    # convert
    pytorch_to_caffe.trans_net(net, input_var, name)
    pytorch_to_caffe.save_prototxt('MGN.prototxt')
    pytorch_to_caffe.save_caffemodel('MGN.caffemodel')
 
    print("Done!")
