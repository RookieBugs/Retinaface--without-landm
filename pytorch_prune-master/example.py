from pruning import create_sparse_table,pruning
import glog
import torchvision.models as models
import torch
from torch.autograd import Variable
vgg16 = models.vgg16(pretrained=True)
input_shape=(1,3,224,224)
sparse_dict_list=create_sparse_table(vgg16,input_shape,DEFAULT_SPARSE_STEP=[0.65, 0.7, 0.75, 0.85, 0.88, 0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99])

label=torch.ones(1,1000)
for total_sparse,sparse_dict in sparse_dict_list:
    glog.info("curr total sparse:{}".format(total_sparse))
    pruning(vgg16,sparse_dict)
    pred=vgg16(Variable(torch.rand(*input_shape)))
    (pred-label).pow(2).mean().backward()

