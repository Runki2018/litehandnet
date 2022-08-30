import matplotlib
import torchvision
import torch
import numpy as np 
import matplotlib.pyplot as plt


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# # For training
# images, boxes = torch.rand(1, 3, 600, 600), torch.tensor([[[2, 2, 50, 50]]])
# labels = torch.randint(1, 91, (4, 1))
# images = list(image for image in images)
# targets = []
# for i in range(len(images)):
#     d = {}     
#     d['boxes'] = boxes[i]
#     d['labels'] = labels[i]
#     targets.append(d)
# output = model(images, targets)
# # print(f"{output.shape=}")
# print(f"{output=}")
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)
# print(f"{predictions=}")

# # optionally, if you want to export the model to ONNX:
# #  torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

lr_list = []
LR = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr = LR)
lambda1 = lambda epoch: np.sin(epoch) / (epoch + 1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)
for epoch in range(100):
    optimizer.step()
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    # lr_list.append(scheduler.get_lr()[0])

plt.plot(range(100),lr_list,color = 'r')
plt.show()

