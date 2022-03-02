from dataclasses import asdict, dataclass
from coqpit import Coqpit, check_argument

@dataclass
class StyleEncoderConfig(Coqpit):
    """Defines the Generic Style Encoder Config

    Args:    
        input_wav (str):
            Path to the wav file used to define the style of the output speech at inference. Defaults to None.

        embedding_dim (int):
            Defines the size of the embedding vector dimensions. Defaults to 256.
        
        gst_style_input_weights (dict):
            Defines the weights for each style token used at inference. Defaults to None.

        gst_num_heads (int):
            Number of attention heads used by the multi-head attention. Defaults to 4.

        gst_num_style_tokens (int):
            Number of style token vectors. Defaults to 10.
    """
    # Style Encoder Type
    se_type: str = "gst"

    # Generic Style Encoder Configuration
    input_wav: str = None
    num_mel: int = 80
    style_embedding_dim: int = 256
    use_speaker_embedding: bool = False

    # GST-Specific Configs
    gst_style_input_weights: dict = None
    gst_num_heads: int = 4
    gst_num_style_tokens: int = 10

    # VAE-SE-Specific Configs
    vae_latent_dim: int = 256

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        super().check_values()
        check_argument("se_type", c, restricted=True, enum_list=["gst", "vae_se"])
        check_argument("input_wav", c, restricted=False)
        check_argument("num_mel", c, restricted=False)
        check_argument("style_embedding_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("use_speaker_embedding", c, restricted=False)
        check_argument("gst_style_input_weights", c, restricted=False)
        check_argument("gst_num_heads", c, restricted=True, min_val=2, max_val=10)
        check_argument("gst_num_style_tokens", c, restricted=True, min_val=1, max_val=1000)
        check_argument("vae_latent_dim", c, restricted=True, min_val=0, max_val=1000)