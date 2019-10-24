
def alignment_diagonal_score(alignments):
    """
    Compute how diagonal alignment predictions are. It is useful
    to measure the alignment consistency of a model
    Args:
        alignments (torch.Tensor): batch of alignments.
    Shape:
        alignments : batch x decoder_steps x encoder_steps
    """
    return alignments.max(dim=1)[0].mean(dim=1).mean(dim=0).item()
