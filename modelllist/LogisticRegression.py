import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LogisticRegression(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_channels, n_classes)
        self.final_act = nn.Softmax(dim=-1)

    def forward(self, batch):
        x = batch.reshape(batch.shape[0], -1)
        # bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        # x = rearrange(x, '(bs e) f -> bs (e f)', bs=bs)
        return self.final_act(self.linear(x))
