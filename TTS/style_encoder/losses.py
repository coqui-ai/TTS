import torch

class DiffusionStyleEncoderLoss(torch.nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        self.config = c
        self.start_loss_at = self.config.start_loss_at
        self.step = 0

        print("Diffusion report - Using alpha loss: ", self.config.diff_loss_alpha)

        if self.config.diff_loss_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.config.diff_loss_type == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError

    def forward(self, diff_output, diff_target):

        if((self.start_loss_at > 0) and (self.step > self.start_loss_at)):
            self.stop_alpha = 1
        else:
            self.stop_alpha = 0

        diff_loss = self.stop_alpha*self.criterion(diff_output, diff_target)
        
        self.step += 1  

        return diff_loss

class VAEStyleEncoderLoss(torch.nn.Module):
    def __init__(self,c) -> None:
        super().__init__()
        self.config = c  
        self.alpha_vae = self.config.vae_loss_alpha # alpha of the loss function, it will be changed while training
        self.step = 0  # it will be incremented every forward and alpha_vae will be recalculated
        self.start_loss_at = self.config.start_loss_at

        print("VAE report - Using Cyclical Annealing: " ,self.config['use_cyclical_annealing'])

    def forward(self, mean, log_var):
        KL = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        
        if((self.start_loss_at > 0) and (self.step > self.start_loss_at)):
            self.stop_alpha = 1
        else:
            self.stop_alpha = 0

        # Doing this here to not affect original training schedules of other style encoders
        if(self.config['use_cyclical_annealing']): 
            # print(self.step, self.alpha_vae) Used to debug, and seems to be working
            self.update_alphavae(self.step)

        self.step += 1  
        
        return self.stop_alpha*KL

    def update_alphavae(self, step):
        self.alpha_vae = min(1, (step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'])
        # Verbose       
        if((step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'] > 1):
            print("VAE: Cyclical annealing restarting") # This print is not working

class VAEFlowStyleEncoderLoss(torch.nn.Module):
    def __init__(self,c) -> None:
        super().__init__()
        self.config = c  
        self.alpha_vae = self.config.vae_loss_alpha # alpha of the loss function, it will be changed while training
        self.step = 0  # it will be incremented every forward and alpha_vae will be recalculated
        self.start_loss_at = self.config.start_loss_at

        print("VAEFlow report - Using Cyclical Annealing: " ,self.config['use_cyclical_annealing'])

    def forward(self, z_0, z_T, mean, log_var):
        
        log_p_z = self.log_Normal_standard(z_T.squeeze(1), dim=1)
        log_q_z = self.log_Normal_diag(z_0.squeeze(1), mean, log_var, dim=1)

        if((self.start_loss_at > 0) and (self.step > self.start_loss_at)):
            self.stop_alpha = 1
        else:
            self.stop_alpha = 0

        KL = (- torch.sum(log_p_z - log_q_z) )

        if(self.config['use_cyclical_annealing']):
            self.update_alphavae(self.step)

        self.step += 1  
        
        return self.stop_alpha*KL

    def update_alphavae(self, step):
        self.alpha_vae = min(1, (step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'])
        # Verbose       
        if((step%self.config['vae_cycle_period'])/self.config['vae_cycle_period'] > 1):
            print("VAE: Cyclical annealing restarting") # This print is not working
        return
        
    def log_Normal_diag(self, x, mean, log_var, average=False, dim=None):
        log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var ), -1) )
        if average:
            return torch.mean(log_normal, dim)
        else:
            return torch.sum(log_normal, dim)

    def log_Normal_standard(self, x, average=False, dim=None):
        log_normal = -0.5 * torch.pow( x , 2 )
        if average:
            return torch.mean(log_normal, dim)
        else:
            return torch.sum(log_normal, dim)
