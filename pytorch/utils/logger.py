improt os
import datetime
import torch.utils.tensorboard import SummaryWriter


class Logger(Functions):
    def __init__(self, path):
        log_dir = os.path.join(
                path, datetime.datetime.now().strftime('%y%m%d%H%M%S'))
        self.writer = SummaryWriter(log_dir)

    def list_of_scalars(self, list_scalar: list, step: int):
        for tag, scalar in list_scaler:
            self.writer.add_scalar(tag, scalar, step)

    def scalar(self, tag, scalar, step: int):
        self.writer.add_scalar(tag, scalar, step)
