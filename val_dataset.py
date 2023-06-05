import os

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):

        assert os.path.exists(root), f"path '{root}' does not exist."

        self.image_root = os.path.join(root, "test", "img")
        self.mask_root = os.path.join(root, "test", "val")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".png")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        re_mask_names = []
        for mask_name in image_names:
            # mask_name = p.replace(".tif", ".png")
            assert mask_name in mask_names, f"{mask_name} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.img_list = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_list = [os.path.join(self.mask_root, n) for n in mask_names]
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.masks_list[idx]).convert('L')
        mask = np.array(mask) / 255
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

if __name__ == '__main__':

    val_dataset = Dataset("../dataset/vaihingen", train=False)
    print(len(val_dataset))

    i, t = val_dataset[0]