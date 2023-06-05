import cv2
from patchify import patchify, unpatchify
import argparse
import os
import numpy as np


def slide_patch(opt):
    large_image = cv2.imread(opt.large_image_dir)
    print("111",large_image)
    padding_h = (((large_image.shape[0] - opt.patch_size[0]) // opt.step + 1) * opt.step + opt.patch_size[0]) - \
                large_image.shape[0]
    padding_w = (((large_image.shape[1] - opt.patch_size[1]) // opt.step + 1) * opt.step + opt.patch_size[1]) - \
                large_image.shape[1]
    large_image_padding = cv2.copyMakeBorder(large_image, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT,
                                             value=(0, 0, 0))
    patches = patchify(large_image_padding, opt.patch_size, opt.step)
    index = 0

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, 0, :, :, :]
            cv2.imwrite(opt.patches_dir + str(index) + '.jpg', single_patch)
            index += 1


def merge_patch(opt):
    large_image = cv2.imread(opt.large_image_dir)
    padding_h = (((large_image.shape[0] - opt.patch_size[0]) // opt.step + 1) * opt.step + opt.patch_size[0]) - \
                large_image.shape[0]
    padding_w = (((large_image.shape[1] - opt.patch_size[1]) // opt.step + 1) * opt.step + opt.patch_size[1]) - \
                large_image.shape[1]
    large_image_padding = cv2.copyMakeBorder(large_image, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT,
                                             value=(0, 0, 0))
    patches = os.listdir(opt.patches_dir)
    patches.sort(key=lambda x: int(x[:x.find('.')]))
    predicted_patches = []
    index = 0
    for i in range((large_image_padding.shape[0] - opt.patch_size[0]) // opt.step + 1):
        for j in range((large_image_padding.shape[1] - opt.patch_size[1]) // opt.step + 1):
            predicted_patch = np.expand_dims(
                np.expand_dims(np.expand_dims(cv2.imread(opt.patches_dir + patches[index]), axis=0), axis=0), axis=0)
            predicted_patches.append(predicted_patch)
            index += 1
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, (
        (large_image_padding.shape[0] - opt.patch_size[0]) // opt.step + 1,
        (large_image_padding.shape[1] - opt.patch_size[1]) // opt.step + 1, 1, opt.patch_size[0], opt.patch_size[1],
        opt.patch_size[2]))
    reconstructed_image = unpatchify(predicted_patches_reshaped, (
        ((large_image_padding.shape[0] - opt.patch_size[0]) // opt.step) * opt.step + opt.patch_size[0],
        ((large_image_padding.shape[1] - opt.patch_size[1]) // opt.step) * opt.step + opt.patch_size[1],
        large_image_padding.shape[2]))
    reconstructed_image_clip = reconstructed_image[:large_image.shape[0], :large_image.shape[1]]
    cv2.imwrite(opt.reconstructed_image_dir, reconstructed_image_clip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=0, help='0 or 1 mean slide or merge')
    parser.add_argument('--large_image_dir', type=str,
                        default='/home/ubuntu/Massachusetts-building/build_test_image/1.tiff',
                        help='large_image path')
    parser.add_argument('--patches_dir', type=str, default='/home/ubuntu/Massachusetts-building/massach/test/',
                        help='save or load patches dir')
    parser.add_argument('--patch_size', type=int, default=(512, 512, 3), help='patch size (height, width, channel)')
    parser.add_argument('--step', type=int, default=256,
                        help='patch slide step, step=224 for 224 patch means no overlap')
    parser.add_argument('--reconstructed_image_dir', type=str, default='C:/User/Desktop/merge.png',
                        help='reconstructed_image save path')
    opt = parser.parse_args()

    if opt.task == 0:
        slide_patch(opt)
    else:
        merge_patch(opt)
