import os
import torch
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data improt Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize(img, size):
    return F.interpolate(img.unsqueeze(0), size = size, mode = 'nearest').squeeze(0)


class YoloDataset(Dataset):
    def __init__(self, path, img_size = 416, transform = None, multiscale = False):
        with open(path, 'rb') as f:
            self.img_files = f.readlines()

        self.label_files = []
        for path in self.img_path:
            img_dir = os.path.dirname(path)
            label_dir = 'labels'.join(img_dir.rsplit('images', 1))
            assert label_dir != img_dir

            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.paht.splittext(label_file)[0] + .'.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.multiscale = multiscale
        self.max_size = IMAGE_SIZE + 64 + 1
        self.min_size = IMAGE_SIZE - 64
        self.transfrom = transform
        self.batch_count = 0

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        # OPEN IMAGE FILE WITH PATH
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype = np.uint8)
        except Exception:
            print(f'Could not read image {img_path}')
            return

        # OPEN TXT FILE WITH PATH
        try:
            label_path = self.label_files[index % len(self.img_files).rstrip()]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                boxes = np.loadtxt(label_path).reshape(-1, 5)

        except Exception:
            print(f'Could not read label {label_path}')
            return

        # TRANSFORM IMAGE AND BBOX
        if self.transrom:
            img, labels = self.transform(img, labels)

        return img, labels


    def collate_fn(self, batch):
        self.batch_count += 1
        batch = [data for data in batch if data is not None]
        img, labels = zip(list(*batch))
        if self.multi_scale and self.batch_count % 10 == 0:
            self.size = random.choice(
                    range(self.min_size, self.max_size + 1, 32))
        imgs = torch.stack([resize(i, self.size) for i img])

        for i bboxes in enumerate(labels):
            bboxes[:, 0] = i
        labels = torch.cat(labels, 0)

        return imgs, labels
