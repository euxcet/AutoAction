import torch
from torch import nn

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, fc_dim, output_dim, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm0 = nn.LSTM(input_dim, hidden_dim, layer_dim,
            batch_first=True, dropout=0.5)
        self.fc0 = nn.Linear(hidden_dim, fc_dim)
        self.fc1 = nn.Linear(fc_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.batch_size = None
        self.hidden = None
        self.device = device

        # init weight
        for name, param in self.lstm0.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else: nn.init.zeros_(param)
        nn.init.xavier_normal_(self.fc0.weight)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)


    def forward(self, x):
        hidden0 = self.init_hidden(x)
        out, hidden1 = self.lstm0(x, hidden0)
        out = self.leaky_relu(self.fc0(out[:, -1, :]))
        out = self.leaky_relu(self.fc1(out))
        return out


    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        if self.device is not None:
            return [t.to(self.device) for t in (h0, c0)]
        return [t for t in (h0, c0)]
