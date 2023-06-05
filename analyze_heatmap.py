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

# create model
model = mpunet1()

weight_path = "save_weights/mpunet1.pth"  # "./resNet34.pth"
# delete weights about aux_classifier
weights_dict = torch.load(weight_path, map_location='cpu')['model']
for k in list(weights_dict.keys()):
    if "aux" in k:
        del weights_dict[k]

# load weights
model.load_state_dict(weights_dict)

# load image
img = Image.open("heatmap/area14.png")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)[0]
print("1111", out_put.type)

feature_map = torch.squeeze(out_put)
feature_map = feature_map.detach().cpu().numpy()

feature_map_sum = feature_map[0, :, :]
feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
for i in range(0, 128):
    feature_map_split = feature_map[i, :, :]
    feature_map_split = np.expand_dims(feature_map_split,axis=2)
    if i > 0:
        feature_map_sum += feature_map_split

    plt.imshow(feature_map_split)
    plt.savefig("./heatmap/" + str(i) + "_{}.jpg".format('drone'))
    plt.xticks()
    plt.yticks()
    plt.axis('off')

plt.imshow(feature_map_sum)
plt.savefig("./heatmap/sum_{}.jpg".format('drone'))
print("save sum_{}.jpg".format('drone'))
