import torch
import torch.nn as nn

config = [
    (32, 3, 1),
    (64, 3, 2),
    [True, 1],
    (128, 3, 2),
    [True, 2],
    (256, 3, 2),
    [True, 8],
    (512, 3, 2),
    [True, 8],
    (1024, 3, 2),
    [True, 4],
    (512, 1, 1),
    (1024, 3, 1),
    'P',
    (256, 1, 1),
    'U',
    (256, 1, 1),
    (512, 3, 1),
    'P',
    (128, 1, 1),
    'U',
    (128, 1, 1),
    (256, 3, 1),
    'P'
]


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, use_bn = True, **kwargs):
        super().___init__()
        self.conv = nn.Conv2d(in_channels = cin,
                              out_channels = cout,
                              bias = not use_bn,
                              **kwargs)
        self.act = nn.LeakyReLU(0.01)
        if use_bn:
            self.bn = nn.BatchNorm2d(cout)

        self.use_bn = use_bn
    
    def forward(self, X):
        if self.use_bn:
            return self.act(self.bn(self.conv(X)))
        else:
            return self.conv(X)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        filters,
        num_iters = 1,
        use_residual = True,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.use_residual = use_residual
        cout = cin // 2
        self.blocks = nn.ModuleList()
        for _ in range(num_iters):
            self.block.append(
                nn.Sequential(
                    ConvBlock(filters, cout, kernel_size = 1, stride = 1, padding = 0),
                    ConvBlock(cout, filters, kernel_size = 3, stride = 1, padding = 1)
                )
            )      
    
    def forward(self, X):
        x = X
        for block in self.blocks:
            if self.use_residual:
                x = block(x) + x
            else:
                x = block(x)
        return x


class Predict(nn.Module):
    def __init__(
        self,
        cin,
        num_classes,
        num_anchors
    ):
        super().__init__()
        cout = cin * 2
        self.sequence = nn.Sequential(
            ConvBlock(cin, cout, kernel_size = 1, stride = 1)
            ConvBlock(cout, (num_classes + 5) * num_anchors, use_bn = False, kernel_size = 1, stride = 1),
        )
        self.num_anchors = num_anchors
        self.num_classes = num_classes
    def forward(self, X):
        return self.sequence(X).reshape(-1, self.num_anchors, self.num_classes + 5, X.shape[2], X.shape[3]).permute(0, 1, 3, 4, 2)


class YOLOv3(nn.Module):
    def __init__(
        self,
        cin,
        num_class,
        num_anchors
    ):
        super().__init__()
        self.cin = cin
        self.num_class = num_class
        self.num_anchors = num_anchors
        self.network = self._make_network()

    def forward(self, X):
        concat = []
        output = []
        x = X
        for idx, layer in enumerate(self.network):
            print(idx, layer.__class__.__name__)
            if isinstance(layer, Predict):
                output.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_iters == 8:
                concat.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([concat[-1], x], dim = 1)
                concat.pop()

        return output

    def _make_network(self):
        network = nn.ModuleList()
        cin = self.cin
        for arch in config:
            if isinstance(arch, tuple):
                cout, kernel_size, stride = arch
                network.append(
                    ConvBlock(
                        cin,
                        cout,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = 1 if kernel_size == 3 else 0
                    )
                )
                cin = cout

            elif isinstance(arch, list):
                residual, num_iters = arch
                network.append(ResidualBlock(cin, num_iters, residual))

            elif isinstance(arch, str):
                if arch == 'U':
                    network.append(nn.Upsample(scale_factor = 2))
                    cin = cin * 3

                elif arch == 'P':
                    network.extend(
                        [ResidualBlock(cin, 1, False),
                         ConvBlock(cin, cin//2, kernel_size = 1, stride = 1),
                         Predict(cin//2, self.num_class, self.num_anchors)]
                    )
                    cin = cin // 2

        return network


if __name__ == '__main__':
    IMAGE_SIZE = 416
    NUM_CHANNEL = 3
    NUM_CLASSES = 80
    NUM_ANCHORS = 3
    model = YOLOv3(NUM_CHANNEL, NUM_CLASSES, NUM_ANCHORS)
    x = torch.randn(1, NUM_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
    outs = model(x)
    for out in outs:
        prtin(out.shape)
