from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGLSTM(nn.Module):

    def __init__(self, num_class=2,
                 num_layer=1,
                 input_channel=4*72,
                 hidden_size=128,
                 dropout=0.3):
        super(EEGLSTM, self).__init__()
        self.num_class = num_class
        self.num_layer = num_layer
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lstm = nn.LSTM(input_channel, hidden_size, num_layer,
                            batch_first=True, bidirectional=False)
        self.predict = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        x, (h, c) = self.lstm(x)
        x = torch.squeeze(x, dim=1)
        x = self.predict(x)
        return x
