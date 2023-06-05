import torch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from src import mpunet1

data_transform = transforms.Compose(
    [transforms.Resize(512),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
model = mpunet1()
# model = resnet34(num_classes=5)
# load model weights
weight_path = "save_weights/mpunet1.pth"  # "./resNet34.pth"
# delete weights about aux_classifier
weights_dict = torch.load(weight_path, map_location='cpu')['model']
for k in list(weights_dict.keys()):
    if "aux" in k:
        del weights_dict[k]

# load weights
model.load_state_dict(weights_dict)
# print(model)

# load image
img = Image.open("heatmap/area14.png")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)[0]
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        plt.savefig('1.png')
        plt.imshow(im[:, :, i], cmap='gray')
    plt.show()

