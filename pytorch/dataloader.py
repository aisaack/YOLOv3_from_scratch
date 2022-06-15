import os
import torch
import numpy as np
import pandas as pd

from utils import iou, nms
from PIL import Image, ImageFile
from torch.utils.data improt Dataset, Dataloader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        label_dir,
        anchors,
        image_size = 416,
        S = [13, 26, 52],
        C = 20,
        transform = None
    ):
        self.annotations = pd.read(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])   # (9, 2) shaped tensor
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // len(S)
        self.C = C
        self.ignore_iou_thresh = .5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])
        image_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        bboxes = np.roll(np.loadtxt(fname = label_path, delimiter = '', ndmin = 2), 4, aixs = 1).tolist()
        iamge = np.array(Image.open(image_path).convert('RGB'))

        if self.transform:
            augementations = self.transform(image = image, bboxes = bboxes)
            image = augementations['image']
            bboxes = augmentations['bboxes']

        targets = [torch.zeors((self.num_anchors_per_scale, len(self.S), S, S, 6)) for S in self.S]    # [p, x, y, w, h, c]
        for bbox in bboxes:
            iou_anchors = iou(torch.tensor(bbox[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending = True, dim = 0)
            x, y, w, h, c = bbox
            has_anchor = [False] * len(self.S)

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale       # which scale?
                anchor_on_scale = anchor_idx % slef.num_anchors_per_scale  # which index?
                S = self.S[scale_idx]
                i, j = int(S * x), int(S * y)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    w_cell, h_cell = w * S, h * S
                    box_coord = torch.tensor([x_cell, y_cell, w_cell, y_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coord
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(c)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchor[anchor_idx] > self.ignore_iou_thresh:    
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(target)
