import torch
import torchvision.models as models


def get_model(model_name='restnet'):  
    resnet50 = models.resnet50(pretrained=False, progress=True, num_classes=10)
    efficentNet_B6 = models.efficientnet_b6(pretrained=False, progress=True, num_classes=10)


# Faster RCNN https://www.cnblogs.com/wildgoose/p/12905004.html