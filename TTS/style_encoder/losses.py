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
        self.step = 0  # it will be incremented every forward and alpha_vae will be recalculated
        
        print("VAE report - Using Cyclical Annealing: " ,self.config['use_cyclical_annealing'])

    def forward(self, mean, log_var):
        KL = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        
        # Doing this here to not affect original training schedules of other style encoders
        if(self.config['use_cyclical_annealing']):
            self.step += 1   
            print(self.step)
            self.update_alphavae(self.step)

        return KL

    def update_alphavae(self, step):
        self.alpha_vae = min(1, (step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'])
        # Verbose       
        if((step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'] > 1):
            print("VAE: Cyclical annealing restarting")