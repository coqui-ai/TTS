import torch
import torch.nn.functional as F
from torch import nn

class VAEFlowStyleEncoder(nn.Module):
    def __init__(self, num_mel, style_emb_dim, latent_dim, intern_dim, number_of_flows):
        super(VAEFlowStyleEncoder, self).__init__()

        self.num_mel = num_mel
        self.style_emb_dim = style_emb_dim
        self.latent_dim = latent_dim
        self.intern_dim = intern_dim
        self.number_of_flows = number_of_flows

        # encoder: q(z | x)
        self.ref_encoder = ReferenceEncoder(num_mel, style_emb_dim)

        self.q_z_layers_pre = nn.ModuleList()
        self.q_z_layers_gate = nn.ModuleList()

        self.q_z_layers_pre.append(nn.Linear(self.style_emb_dim, self.intern_dim))
        self.q_z_layers_gate.append(nn.Linear(self.style_emb_dim, self.intern_dim))

        self.q_z_layers_pre.append(nn.Linear(self.intern_dim, self.intern_dim))
        self.q_z_layers_gate.append(nn.Linear(self.intern_dim, self.intern_dim))

        self.q_z_mean = nn.Linear(self.intern_dim, self.latent_dim)
        self.q_z_logvar = nn.Linear(self.intern_dim, self.latent_dim)

        # Householder flow
        self.v_layers = nn.ModuleList()
        # T > 0
        if self.number_of_flows > 0:
            # T = 1
            self.v_layers.append(nn.Linear(intern_dim, self.latent_dim))
            # T > 1
            for i in range(1, self.number_of_flows):
                self.v_layers.append(nn.Linear(self.latent_dim, self.latent_dim))

        self.sigmoid = nn.Sigmoid()
        self.Gate = Gate()
        self.HF = HF()

        # Xavier initialization (normal)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) # Good init for projection

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        h0_pre = self.q_z_layers_pre[0](x)
        h0_gate = self.sigmoid( self.q_z_layers_gate[0](x) )
        h0 = self.Gate( h0_pre, h0_gate )

        h1_pre = self.q_z_layers_pre[1](h0)
        h1_gate = self.sigmoid( self.q_z_layers_gate[1](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        z_q_mean = self.q_z_mean(h1)
        z_q_logvar = self.q_z_logvar(h1)
        return z_q_mean, z_q_logvar, h1

    # THE MODEL: HOUSEHOLDER FLOW
    def q_z_Flow(self, z, h_last):
        v = {}
        # Householder Flow:
        if self.number_of_flows > 0:
            v['1'] = self.v_layers[0](h_last)
            z['1'] = self.HF(v['1'], z['0'])
            for i in range(1, self.number_of_flows):
                v[str(i + 1)] = self.v_layers[i](v[str(i)])
                z[str(i + 1)] = self.HF(v[str(i + 1)], z[str(i)])
        return z

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        
        # Ref Encoder Pass
        x = self.ref_encoder(x)
        
        # VAE Pass (z ~ q(z | x))
        z_q_mean, z_q_logvar, h_last = self.q_z(x)
        z = {}
        z['0'] = self.reparametrize(z_q_mean, z_q_logvar)

        # Householder Flow Pass:
        z = self.q_z_Flow(z, h_last)
        return {'z' : z[str(self.number_of_flows)].unsqueeze_(1) , 'z_0':z['0'] , 'mean': z_q_mean, 'log_var': z_q_logvar, 'out': x} #out will be the RE output, just to have the output if needed

class HF(nn.Module):
    def __init__(self):
        super(HF, self).__init__()

    def forward(self, v, z):
        '''
        :param v: batch_size (B) x latent_size (L)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = z - 2* v v_T / norm(v,2) * z
        '''
        # v * v_T
        vvT = torch.bmm( v.unsqueeze(2), v.unsqueeze(1) )  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L
        # v * v_T * z
        vvTz = torch.bmm( vvT, z.unsqueeze(2) ).squeeze(2) # A * z : batchdot( B x L x L * B x L x 1 ).squeeze(2) = (B x L x 1).squeeze(2) = B x L
        # calculate norm ||v||^2
        norm_sq = torch.sum( v * v, 1 ).unsqueeze_(1) # calculate norm-2 for each row : B x 1
        norm_sq = norm_sq.expand( norm_sq.size(0), v.size(1) ) # expand sizes : B x L
        # calculate new z
        z_new = z - 2 * vvTz / norm_sq # z - 2 * v * v_T  * z / norm2(v)
        return z_new

class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g

###########################
#### Reference Encoder ####
###########################

class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.
    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim, batch_first=True
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)            # [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
        x = x.transpose(1, 2)                                       # [batch_size, post_conv_width,num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)    # [batch_size, post_conv_width,num_channels*post_conv_height]
        self.recurrence.flatten_parameters()
        _, out = self.recurrence(x)                                 # [seq_len==1, batch_size, encoding_size=128]
        return out.squeeze(0)

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad, n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height