# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu


class GLU(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = _get_activation_fn(activation=str(activation_fn))
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.gate.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.fc2(x)
        x = x.view(x_shape)
        return x
