import random
import numpy as np

import torch
from torchvision import transforms
import albumentations as A

from config import IMAGE_SIZE, TRAIN_TRANSFORM, TEST_TRANSFORM


class AbsoluteCoord(object):
    def __call__(self, data):
        img, label = data
        ih, iw = img.shape[:2]

        label[..., [1, 3]] *= iw
        label[..., [2, 4]] *= ih

        return img, label


class RelativeCoord(object):
    def __call__(self, data):
        img, label = data
        ih, iw = img.shape[:2]

        label[..., [1, 3]] /= iw
        label[..., [2, 4]] /= ih

        return img, label


class PadIfNeeded(object):
    def __init__(self, size):
        self.pad = A.PadIfNeeded(
                min_height = size,
                min_width = size,
                border_mode = 0
                )

    def __call__(self, data):
        img, label = data
        aug = self.pad(image = img, bboxes = label[..., 1:])
        img = aug['image']
        label[..., 1:] = aug['bboxes']

        return img, label


class ToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, data):
        img, label = data
        img = self.totensor(img)
        bb_target = torch.zeros((len(label), 6))
        bb_target[..., 1:] = torch.from_numpy(label)

        return img, bb_target


class Albumentation:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        img, label = data
        label[..., 1:] = self._absolutecoord(img, label[..., 1:])
        label[..., 1:] = self._xywh2xyxy(label[..., 1:])
        label[..., 1:] = self._xyxy2xywh(img, label[..., 1:])
        label[..., 1:] = self._relativecoord(img, label[..., 1:])
        aug = self.transform(image = img, bboxes = label[..., 1:])
        img = aug['image']
        label[..., 1:] = aug['bboxes']

        return img, label


    def _xyxy2xywh(self, img, bbox):
        x1, y1, x2, y2 = bbox.T
        ih, iw = img.shape[:2]
        bbox[..., 0] = np.maximum((x2 - x1) / 2, 0)
        bbox[..., 1] = np.maximum((y2 - y1) / 2, 0)
        bbox[..., 2] = np.minimum((x2 - x1) + 0.1, iw)
        bbox[..., 3] = np.minimum((y2 - y1) + 0.1, ih)

        return img, bbox

    def _xywh2xyxy(self, bbox):
        x, y, w, h = bbox.T
        bbox[..., 0] = x - w / 2
        bbox[..., 1] = y - h / 2
        bbox[..., 2] = x + w / 2
        bbox[..., 3] = y + h / 2

        return bbox

    def _absolutecoord(self, img, bbox):
        ih, iw = img.shape[:2]
        bbox[..., [0, 2]] *= iw
        bbox[..., [1, 3]] *= ih

        return bbox.astype(int)

    def _relativecoord(self, img, bbox):
        ih, iw = img.shape[:2]
        bbox[..., [0, 2]] /= iw
        bbox[..., [1, 3]] /= ih

        return bbox


train_transform = transforms.Compose([
    Albumentation(TRAIN_TRANSFORM),
    AbsoluteCoord(),
    PadIfNeeded(IMAGE_SIZE),
    RelateveCoord(),
    ToTensor()])


test_transform = transforms.Compose([
    Albumentation(TEST_TRANSFORM),
    AbsoluteCoord(),
    PadIfNeeded(IMAGE_SIZE),
    RelateveCoord(),
    ToTensor()])

