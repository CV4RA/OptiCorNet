import torch
import torch.nn as nn
import numpy as np

class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):
        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w

        # 1D convolution
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)  # [B,C] -> [B,1,C]

        x = x.permute(0, 2, 1)  # [B,T,C] -> [B,C,T]
        seqFt = self.conv(x)    # [B, outDims, T']
        seqFt = torch.mean(seqFt, dim=-1)  # [B, outDims]

        return seqFt


class Delta(nn.Module):
    def __init__(self, inDims, seqL, outDims=None, lstm_layers=1, use_residual=True):
        super(Delta, self).__init__()
        self.inDims = inDims
        self.seqL = seqL
        self.use_residual = use_residual

        # differential weight
        weight = (np.ones(seqL, np.float32)) / (seqL / 2.0)
        weight[:seqL // 2] *= -1
        self.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)

        self.outDims = outDims if outDims else inDims

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=inDims,
            hidden_size=self.outDims,
            num_layers=lstm_layers,
            batch_first=True
        )

        # projection layer
        if self.use_residual and self.outDims != inDims:
            self.proj = nn.Linear(inDims, self.outDims)
        else:
            self.proj = None

    def forward(self, x):
        # input x: [B, T, C]
        x = x.permute(0, 2, 1)  # [B, C, T]
        delta = torch.matmul(x, self.weight)  # [B, C]
        delta_exp = delta.unsqueeze(1)        # [B, 1, C]

        # LSTM feedforward
        lstm_out, _ = self.lstm(delta_exp)    # [B, 1, outDims]

        if self.use_residual:
            residual = delta  # [B, C]
            if self.proj:
                residual = self.proj(residual)  # [B, outDims]
            out = lstm_out.squeeze(1) + residual  # [B, outDims]
        else:
            out = lstm_out.squeeze(1)  # [B, outDims]

        return out