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

class VAEStyleEncoderLoss(torch.nn.Module):
    def __init__(self,c) -> None:
        super().__init__()
        self.config = c  
        self.alpha_vae = 1.0 # alpha of the loss function, it will be changed while training

        def forward(self, mean, log_var):
            KL = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
            return KL