import torch

class DiffusionStyleEncoderLoss(torch.nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        self.config = c
        
        if self.config.diff_loss_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.config.diff_loss_type == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError

    def forward(self, diff_output, diff_target):
        diff_loss = self.criterion(diff_output, diff_target)
        return diff_loss