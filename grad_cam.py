import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from src import mpunetx
import torch
import torch.functional as F
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2


image_url = "area14.png"
image = np.array(Image.open(image_url))
rgb_img = np.float32(image) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


# create model
model = mpunetx()

weight_path = "/home/ubuntu/zhao/deep-learning/pytorch_segmentation/mpunet/save_weights/mpunet1.pth"  # "./resNet34.pth"
# delete weights about aux_classifier
weights_dict = torch.load(weight_path, map_location='cpu')['model']
for k in list(weights_dict.keys()):
    if "aux" in k:
        del weights_dict[k]

# load weights
model.load_state_dict(weights_dict)

model = model.eval().cpu()

# if torch.cuda.is_available():
    # model = model.cuda()
    # input_tensor = input_tensor.cuda()

output = model(input_tensor)
# print(type(output), output.keys())

normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
sem_classes = [
    '__background__', 'building'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

car_category = sem_class_to_idx["building"]
car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
car_mask_float = np.float32(car_mask == car_category)

both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
Image.fromarray(both_images)

from pytorch_grad_cam import GradCAM


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        # if torch.cuda.is_available():
        #     self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


target_layers = [model.cbam4]
# target_layers = [model.out_conv]
targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    print(cam_image.shape)
    cv2.imwrite("cbamTK_cbam4.png", cam_image)
Image.fromarray(cam_image)
