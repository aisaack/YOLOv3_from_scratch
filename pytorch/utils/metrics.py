
def _get_grid(p):
    bn, _, _, gy, gx = p.size()
    vy, vx = torch.meshgrid(torch.arange(gy), torch.arange(gx), indexing = 'ij')
    grid = torch.stack([vx, vy], dim = 1).view(1, 1, gx, gy, 2)
    return bn, grid

def _build_output(pred, model):
    i = 0 
    ps = []
    for m in model.module.layers():
        if isinstance(m, YoloLayer):
            anchor = m.anchor
            stride = m.stride
            p = pred[i]
            nb, grid = _get_grid(p)
            i += 1
            p[..., :2] = (torch.sigmoid(p[..., :2]) + grid) * stride
            p[..., 2:4] = torch.exp(p[..., 2:4]) * anchor.view(1, -1, 1, 1, 2)
            p[..., 4:] = torch.sigmoid(x[..., 4:])
            ps.append(p.view(nb, -1, NUM_ATTR))
    return torch.cat(ps, 1)

def nms(self, pred):
    pred = _build_output(pred)

    # IMPLEMENT NMS
