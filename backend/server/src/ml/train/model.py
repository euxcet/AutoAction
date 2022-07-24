import torch
from torch import device, nn
import numpy as np

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, fc_dim, output_dim, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm0 = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) # delete dropout
        self.fc0 = nn.Linear(hidden_dim, fc_dim)
        self.fc1 = nn.Linear(fc_dim, output_dim)
        self.batch_size = None
        self.hidden = None
        self.device = device


    def forward(self, x):
        hidden0 = self.init_hidden(x)
        out, hidden1 = self.lstm0(x, hidden0)
        out = self.fc0(out[:, -1, :])
        out = self.fc1(out)
        return out


    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        if self.device is not None:
            return [t.to(self.device) for t in (h0, c0)]
        return [t for t in (h0, c0)]


class ConvBnBlock(nn.Module):
    ''' A convolution followed by batch normalization and an activation function.

    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 groups=1, bias=False, batch_normalization=False, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=False) if batch_normalization else nn.Identity()
        self.activation = nn.SiLU() if activation else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SEBlock(nn.Module):
    ''' Squeeze-and-Excitation block assigns weights to the kernels.
    '''
    def __init__(self, channel, reduction=24):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(channel, channel // reduction, kernel_size=1),
                                        nn.SiLU(),
                                        nn.Conv2d(channel // reduction, channel, kernel_size=1),
                                        nn.Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(x)
        return x * y


class MBConvN(nn.Module):
    ''' MBConvN means an MBConv with an expansion factor of n.
    '''
    def __init__(self, in_channels, out_channels, expansion, kernel_size=3, stride=1, reduction=24):
        super().__init__()
        padding = (kernel_size - 1) // 2
        expanded_channel = in_channels * expansion
        self.skip_connection = in_channels == out_channels and stride == 1
        self.expand = ConvBnBlock(in_channels, expanded_channel, kernel_size=1) if (expansion != 1) else nn.Identity()
        self.depthwise = ConvBnBlock(expanded_channel, expanded_channel, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=expanded_channel)
        self.se = SEBlock(expanded_channel, reduction=reduction)
        self.reduce = ConvBnBlock(expanded_channel, out_channels, kernel_size=1, activation=False)
        # do drop?

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reduce(x)
        if self.skip_connection:
            x = x + residual
        return x

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                groups=in_channels, bias=bias, padding=1, stride=stride)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CNNClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        # shared parameters for each axis
        self.MBConv1 = MBConvN(4, 16, 1, reduction=2)
        self.MBConv6 = MBConvN(16, 24, 6, stride=2, reduction=1)
        self.SepConv = SeparableConv(100, 41, 3, 6)
        self.MaxPool = nn.MaxPool2d((3, 3), stride=(8, 1), padding=(0, 1))
        self.Dense0 = nn.Linear(180, 120)
        self.Dense1 = nn.Linear(120, 60)
        self.Dense2 = nn.Linear(60, 30)
        self.Dense3 = nn.Linear(30, 10)

    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.unsqueeze(-1)
        ys = []
        for i in range(0, 9):
            y = x[:, i * 4: i * 4 + 4, :, :]
            y = self.MBConv1(y)
            y = self.MBConv6(y)
            ys.append(y)
        z = torch.cat(ys, 1)
        z = z.squeeze(dim=3)
        z = z.transpose(1, 2)
        z = self.SepConv(z)
        z = self.MaxPool(z)
        z = torch.flatten(z, 1, 2)
        z = self.Dense0(z)
        z = self.Dense1(z)
        z = self.Dense2(z)
        z = self.Dense3(z)
        return z

if __name__ == '__main__':
    '''
    input       [bs, 200, 36]
    view        [bs, 200, 9, 4]
    split       9 * [bs, 200, 1, 4]
    transpose   9 * [bs, 4, 200, 1]
    MBConv1     9 * [bs, 16, 100, 1]
    MBConv2     9 * [bs, 24, 50, 1]
    concat      [bs, 216, 50, 1]
    view        [bs, 50, 216]
    seperable   [bs, 41, 36]
    maxpool     [bs, 5, 36]
    flatten     [bs, 180]
    denses      [bs, 10]
    '''
    # unit tests for cnn
    # 32  4 * 9  200  1
    x = np.ones((32, 200, 36), dtype=np.float32)
    x = torch.tensor(x)
    # model = MBConvN(4, 16, 1, reduction=2)
    model = CNNClassifier()
    y = model(x)
    print('y', y.shape)
