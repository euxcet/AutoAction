import torch
from torch import device, nn

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, fc_dim, output_dim, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm0 = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) # delete dropout
        self.fc0 = nn.Linear(hidden_dim, fc_dim        self.fc1 = nn.Linear(fc_dim, output_dim)
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
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0,
                 groups=1, bias=False, batch_normalization=True, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel) if batch_normalization else nn.Identitiy()
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
    def __init__(self, in_channel, out_channel, expansion, kernel_size=3, stride=1, reduction=24):
        super().__init__()
        padding = (kernel_size - 1) // 2
        expanded_channel = in_channel * expansion
        self.skip_connection = in_channel == out_channel and stride == 1
        self.expand = ConvBnBlock(in_channel, expand_channel, kernel_size=1) if (expansion == 1) else nn.Identity()
        self.depthwise = ConvBnBlock(expanded_channel, expanded_channel, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=expanded_channel)
        self.se = SEBlock(expanded_channel, reduction=reduction)
        self.reduce = ConvBnBlock(expanded_channel, out_channel, kernel_size=1, activation=False)
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

class CNNClassifier(nn.Module):

    def __init__(self, height, width, output_dim, device=None):
        super().__init__()
        self.device = device
    
    def forward(self, x):
        pass

if __name__ == '__main__':
    # unit tests for cnn