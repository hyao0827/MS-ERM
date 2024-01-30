
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.k = configs.top_k
        self.conv1 = nn.Conv2d(in_channels=47, out_channels=1024, kernel_size=(1, 9), padding=(0, 4), stride=1)
        self.conv2 = nn.Conv2d(in_channels=47, out_channels=1024, kernel_size=(1, 7), padding=(0, 3), stride=1)
        self.conv3 = nn.Conv2d(in_channels=47, out_channels=1024, kernel_size=(1, 5), padding=(0, 2), stride=1)
        self.BN_t = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=47, kernel_size=(5, 1), padding=(2, 0), stride=1)
        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=47, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.BN_s = nn.BatchNorm2d(47)

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - T), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()

            # 2D conv: from 1d Variation to 2d Variation
            out1 = self.conv1(out)
            out1 = self.relu(out1)
            out2 = self.conv2(out)
            out2 = self.relu(out2)
            out_ = torch.cat((out2, out1), dim=-1)
            out3 = self.conv3(out)
            out3 = self.relu(out3)
            out = torch.cat((out_, out3), dim=-1)
            out_ = self.BN_t(out)

            out4 = self.conv4(out_)
            out4 = self.relu(out4)
            out5 = self.conv5(out_)
            out5 = self.relu(out5)
            out__ = torch.cat((out4, out5), dim=2)
            output = self.BN_s(out__)

            # reshape back
            output = output.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(output[:, :T, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class MT_timenets(nn.Module):
    def __init__(self, configs):
        super(MT_timenets, self).__init__()
        self.configs = configs
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(47)
        self.act = F.gelu
        self.dropout = nn.Dropout(0.8)
        self.projection = nn.Linear(4*47, 2)

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 4

        self.Tception1 = self.conv_block(1, 15, (1, 16), 1,  self.pool)
        self.Tception2 = self.conv_block(1, 15, (1, 8), 1,  self.pool)
        self.Tception3 = self.conv_block(1, 15, (1, 4), 1,  self.pool)
        self.Sception1 = self.conv_block(15, 1, (4, 1), 1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(15, 1, (2, 1), 1, int(self.pool * 0.25))
        self.BN_t = nn.BatchNorm2d(15)
        self.BN_s = nn.BatchNorm2d(1)
        self.BN_fusion = nn.BatchNorm2d(1)

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)

        out = out.squeeze(dim=1)

        # TimesNet
        for i in range(self.layer):
            x = self.layer_norm(self.model[i](out))
        output = self.act(x)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))
