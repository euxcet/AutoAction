import torch
from torch import nn

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, use_cuda=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm0 = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)
        self.fc0 = nn.Linear(hidden_dim, 32)
        self.fc1 = nn.Linear(32, output_dim)
        self.batch_size = None
        self.hidden = None
        self.use_cuda = use_cuda


    def forward(self, x):
        hidden0 = self.init_hidden(x)
        out, hidden1 = self.lstm0(x, hidden0)
        out = self.fc0(out[:, -1, :])
        out = self.fc1(out)
        return out


    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        if self.use_cuda:
            return [t.cuda() for t in (h0, c0)]
        return [t for t in (h0, c0)]
