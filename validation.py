import os
import torch

from src import mpunetx
from train_utils import evaluate
from val_dataset import Dataset
import transforms as T

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # segmentation nun_classes + background
    num_classes = args.num_classes

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = Dataset(args.data_path,
                          train=True,
                          transforms=SegmentationPresetEval(512),
                          )

    num_workers = min([os.cpu_count(), 2])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = mpunetx()
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
    print(confmat)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch cad validation")

    parser.add_argument("--data-path", default="../dataset/vaihingen", help="Data root")
    parser.add_argument("--weights", default="./save_weights/model_194.pth")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
