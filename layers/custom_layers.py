# coding: utf-8
# import torch
# from torch import nn

# class StopProjection(nn.Module):
#     r""" Simple projection layer to predict the "stop token"

#     Args:
#         in_features (int): size of the input vector
#         out_features (int or list): size of each output vector. aka number
#             of predicted frames.
#     """

#     def __init__(self, in_features, out_features):
#         super(StopProjection, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.dropout = nn.Dropout(0.5)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, inputs):
#         out = self.dropout(inputs)
#         out = self.linear(out)
#         out = self.sigmoid(out)
#         return out
