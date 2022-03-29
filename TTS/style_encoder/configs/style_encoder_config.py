from dataclasses import asdict, dataclass
from coqpit import Coqpit, check_argument

@dataclass
class StyleEncoderConfig(Coqpit):
    """Defines the Generic Style Encoder Config

    Args:    
        embedding_dim (int):
            Defines the size of the embedding vector dimensions. Defaults to 256.
        
        gst_style_input_weights (dict):
            Defines the weights for each style token used at inference. Defaults to None.

        gst_num_heads (int):
            Number of attention heads used by the multi-head attention. Defaults to 4.

        gst_num_style_tokens (int):
            Number of style token vectors. Defaults to 10.

        se_type (str):
            Style encoder type \in {diffusion, vae, gst, re}
    """
    # Style Encoder Type
    se_type: str = "diffusion"

    # Generic Style Encoder Configuration
    num_mel: int = 80
    style_embedding_dim: int = 128
    use_speaker_embedding: bool = False

    # GST-Specific Configs
    gst_style_input_weights: dict = None
    gst_num_heads: int = 4
    gst_num_style_tokens: int = 10

    # VAE-SE-Specific Configs
    vae_latent_dim: int = 128 # Dim of mean and logvar
    embedding_dim: int = 128 # Dim of reference encoder output
    use_cyclical_annealing: bool = True # Whether use or not annealing (recommended true), only linear implemented
    vae_loss_alpha: int = 1.0 # Default alpha value (term of KL loss)
    vae_cycle_period: int = 5000 # iteration period to apply a new annealing cycle

    # Diffusion-specific Configs
    diff_num_timesteps: int = 25 
    diff_schedule_type: str = 'cosine'
    diff_loss_type: str = 'l1' 
    diff_ref_online: bool = True 
    diff_step_dim: int = 128
    diff_in_out_ch: int = 1 
    diff_num_heads: int = 1 
    diff_hidden_channels: int = 128 
    diff_num_blocks: int = 5
    diff_dropout: int = 0.1
    diff_loss_alpha: int = 0.75

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        super().check_values()
        check_argument("se_type", c, restricted=True, enum_list=["gst", "vae", "diffusion"])
        check_argument("num_mel", c, restricted=False)
        check_argument("style_embedding_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("use_speaker_embedding", c, restricted=False)
        check_argument("gst_style_input_weights", c, restricted=False)
        check_argument("gst_num_heads", c, restricted=True, min_val=2, max_val=10)
        check_argument("gst_num_style_tokens", c, restricted=True, min_val=1, max_val=1000)
        check_argument("vae_latent_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("diff_num_timesteps", c, restricted=True, min_val=0, max_val=5000)
        check_argument("diff_schedule_type", c, restricted=True, enum_list=["cosine", "linear"])
        check_argument("diff__step", c, restricted=True, min_val=0, max_val=self.num_timesteps)
        check_argument("diff_loss_type", c, restricted=True, enum_list=["l1", "mse"])
        check_argument("diff_ref_online", c, restricted=True, enum_list=[True, False])
        check_argument("diff_step_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("diff_in_out_ch", c, restricted=True, min_val=1, max_val=1)
        check_argument("diff_num_heads", c, restricted=False)
        check_argument("diff_hidden_channels", c, restricted=True, min_val=0, max_val=2048)
        check_argument("diff_num_blocks", c, restricted=True, min_val=0, max_val=100)
        check_argument("diff_dropout", c, restricted=False)
        check_argument("diff_loss_alpha", c, restricted=False)





