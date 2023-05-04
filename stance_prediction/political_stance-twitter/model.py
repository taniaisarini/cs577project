import torch
class LSTM_Model(torch.nn.Module):
    def __init__(self, num_embedding, embedding_dim, pretrained=None):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_embedding, embedding_dim)
        if pretrained is not None:
            self.embeddings = torch.nn.Embedding.from_pretrained(pretrained)
        self.layer1 = torch.nn.LSTM(embedding_dim, 10, batch_first=True)
        self.layer2 = torch.nn.Linear(10, 1)
    def forward(self, X):
#         print(X)
        X, (_, _) = self.layer1(self.embeddings(X))
        X = torch.nn.ReLU()(X[:, -1])
        X = torch.nn.Sigmoid()(self.layer2(X))
#         print(X.size())
        return X