import torch

class DAN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_size):
        super(DAN, self).__init__()

        self.linear1 = torch.nn.Linear(vocab_size, hidden_dim)

        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, 64)
        self.linear3 = torch.nn.Linear(64, output_size)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x


class RNN(torch.nn.Module):
    def __init__(self, vocab_size, output_size, hidden_dim, n_layer, bidirectional, seqLength, embs_dim=50):
        # TODO: Declare RNN model architecture
        super(RNN, self).__init__()
        self.embs_dim = embs_dim
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.d = 1
        if bidirectional is True:
            self.d = 2

        self.seqLength = seqLength

        self.embedding = torch.nn.Embedding(vocab_size + 1, embs_dim)
        self.rnn = torch.nn.RNN(input_size=embs_dim, hidden_size=hidden_dim, num_layers=n_layer, batch_first=True, bidirectional=bidirectional)
        self.linear1 = torch.nn.Linear(hidden_dim * self.d, 64)
        self.linear2 = torch.nn.Linear(64, output_size)
        self.relu = torch.nn.ReLU()
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.rnn(x)

        x = x[[i for i in range(x.shape[0])], x.shape[1] - 1, :]

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

