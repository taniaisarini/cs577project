import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import utils
from nn_arch import LSTM

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

filename = 'News_Category_Dataset_v3.json'
embedding_size_ = 50
learning_rate_ = 0.01
batch_size_ = 100


def accuracy(x, y):
    count = 0
    for x_, y_ in zip(x,y):
        if (x_ == y_):
            count += 1
    return count


def train(model, loss, optimizer, dataloader):
    model.train()
    train_loss = 0
    correct = 0
    for label, x in dataloader:
        optimizer.zero_grad()
        output = model.forward(x.detach())
        # Using torchhe output of last element in sequence as predicted value
        result = output[:, -1:, :].to(torch.float).squeeze(1)
        loss_calc = loss(result, label)
        train_loss += loss_calc.item()
        loss_calc.backward()
        optimizer.step()
        correct += accuracy(result.argmax(dim=1), label.argmax(dim=1))
    return correct, train_loss

def validate(model, loss, dataloader):
    model.eval()
    train_loss = 0
    correct = 0
    for label, x in dataloader:
        output = model.forward(x.detach())
        # Using torchhe output of last element in sequence as predicted value
        result = output[:, -1:, :].to(torch.float).squeeze(1)
        loss_calc = loss(result, label)
        train_loss += loss_calc.item()
        loss_calc.backward()
        correct += accuracy(result.detach().argmax(dim=1), label.argmax(dim=1))
    return correct, train_loss

def main():
    dataset = utils.NewsDataset('category_data.json')
    train_set, val_set = torch.utils.data.random_split(dataset, [1000, 500])
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_)
    dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=batch_size_)

    model = LSTM(input_size=embedding_size_,
                 output_size=dataset.num_categories,
                 hidden_size=embedding_size_,
                 num_layers=1,
                 bidirectional=False,
                 num_epochs=1).to(torch_device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_)
    for epoch in range(1000):
        if epoch % 10 == 0:
            correct_train, train_loss = train(model, loss, optimizer, dataloader)
            correct_val, val_loss = validate(model, loss, dataloader_val)
            print("******* epoch {} *******".format(epoch))
            print("training accuracy: {}".format(correct_train / dataset.__len__()))
            print("training loss: {}".format(train_loss / dataset.__len__()))
            print("validation accuracy: {}".format(correct_val / val_set.__len__()))
            print("validation loss: {}".format(val_loss / val_set.__len__()))


if __name__ == "__main__":
    # testing

    main()
