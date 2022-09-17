import torch
import torch.nn as nn

from config NUM_ATTR, NUM_ANCHOR_PER_SCALE


class Block(nn.Module):
    def __init__(self, cin, cout, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
                in_channels = cin,
                out_channels = cout,
                **kwargs)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeackyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, cout):
        cinter = cout // 2
        self.block1 = Block(cout, cinter, kernel_size = 1)
        self.block2 = Block(cinter, cout, kernel_size = 3, padding = 1)

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        return x + residual

class Darknet53(nn.Module):
    def __init__(self, cin):
        self.stem = Block(cin, 32, kernel_size = 1)
        self.satge1 = self._create_stage(32, 64, 1)
        self.satge2 = self._create_stage(64, 128, 2)
        self.satge3 = self._create_stage(128, 256, 8)
        self.satge4 = self._create_stage(256, 512, 8)
        self.satge5 = self._create_stage(512, 1024, 4)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        sml = self.stage3(x)
        mid = self.stage4(x)
        lrg = self.stage5(x)
        return sml, mid, lrg

    def _create_stage(cin, cout, num_iter):
        stage = nn.Sequential()
        stage.append(Block(cin, cout, kernel_size = 3, stride = 2, padding = 1))
        for _ in range(num_iter):
            stage.append(ResBlock(cout))
        return stage


class YoloLayer(nn.Module):
    def __init__(self, scale, stride):
        super().__init__()
        S = {'s': [0, 1, 2], 'm': [3, 4, 5], 'l': [6, 7, 8]}
        self.acnhor = torch.tensor([ANCHOR[s] for s in S[scale]])
        self.stride = stride

    def forward(self, x):
        num_batch, _, gy, gx = x.size()
        x = x.view(
                num_batch, NUM_ANCHOR_PER_SCALE,
                NUM_ATTR, num_grid, num_grid).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:
            vy, vx = torch.meshgrid(torch.arange(gy), torch.arange(gx), indexing = 'ij')
            grid = torch.stack([vx, vy], dim = 1).view((1, 1, gx, gy, 2)).float()
            x[..., :2] = (torch.sigmoid(x[..., :2]) + grid) * self.stride
            x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor.view(1, -1, 1, 1, 2)
            x[..., 4:] = torch.sigmoid(x[..., 4:])
            x = x.view(num_batch, -1 NUM_ATTR)

        return x


class PredBlock(nn.Module):
    def __init__(self, cin, cout, scale, stride):
        super().__init__()
        cinter = cout // 2
        self.block1 = Block(cin, cinter, kernel_size = 1)
        self.block1 = Block(cinter, cout, kernel_size = 3, padding = 1)
        self.block1 = Block(cout, cinter, kernel_size = 1)
        self.block1 = Block(cinter, cout, kernel_size = 3, padding = 1)
        self.block1 = Block(cout, cinter, kernel_size = 1)
        self.block1 = Block(cinter, cout, kernel_size = 3, padding = 1)
        self.block1 = Block(cout, NUM_ATTR * NUM_ANCHOR_PAER_SALE, kernel_size = 1)
        self.yolo_layer = YoloLayer(scale, stride)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        self.branch = self.block5(x)
        x = self.block6(self.branch)
        x = self.block7(x)
        x = self.yolo_layer(x)
        return x


class YoloHead(nn.Module):
    def __init__(self):
        self.detect1 = PredBlock(1024, 1024, 'l', 32)
        self.conv1 = Block(512, 256, kernel_size = 1)
        self.detect2 = PredBlock(768, 512, 'm', 16)
        self.conv2 = Block(256, 128, kernel_size = 1)
        self.detect3 = PredBlock(364, 256, 's', 8)

    def forward(self, x1, x2, x3):
        lrg = self.detect1(x1)
        branch = self.detect1.branch
        x = self.conv1(branch)
        x = F.interpolate(x, scale_factor = 2)
        x = torch.cat([x, x2], 1)
        mid = self.detect1(x)
        branch = self.detect1.branch
        x = self.conv2(branch)
        x = F.interpolate(x, scale_factor = 2)
        x = torch.cat([x, x3], 1)
        sml = self.detect3(x)
        return lrg, mid, sml


class Yolov3(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.backbone = Darknet53(cin)
        self.yolo_head = YoloHead()

    def forward(self, x):
        sml, mid, lrg = self.backbone(x)
        lrg, mid, sml = self.yolo_head(lrg, mid, sml)
        out = [lrg, mid, sml]

        if not self.training:
            out = torch.cat(out, 1)

        return out

