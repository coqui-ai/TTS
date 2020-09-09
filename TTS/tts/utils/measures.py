def alignment_diagonal_score(alignments, binary=False):
    """
    Compute how diagonal alignment predictions are. It is useful
    to measure the alignment consistency of a model
    Args:
        alignments (torch.Tensor): batch of alignments.
        binary (bool): if True, ignore scores and consider attention
        as a binary mask.
    Shape:
        alignments : batch x decoder_steps x encoder_steps
    """
    maxs = alignments.max(dim=1)[0]
    if binary:
        maxs[maxs > 0] = 1
    return maxs.mean(dim=1).mean(dim=0).item()
