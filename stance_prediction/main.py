import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import utils
from nn_arch import LSTM

from datasets import load_dataset
import json

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

filename = 'News_Category_Dataset_v3.json'
embedding_size_ = 50
learning_rate_ = 0.0001
batch_size_ = 1


def predict(x):
    if x > 0.5:
        return 1
    else:
        return 0


def accuracy(x, y):
    count = 0
    # for x_, y_ in zip(x,y):
    #     if (x_ == y_):
    #         count += 1
    if x == y:
        count += 1
    return count


def train(model, loss, optimizer, dataloader):
    model.train()
    train_loss = 0
    correct = 0
    for x, label in dataloader:
        optimizer.zero_grad()
        output = model.forward(x.detach())
        # Using torchhe output of last element in sequence as predicted value
        result = output[:, -1:, :].to(torch.float).flatten()
        loss_calc = loss(result, label.flatten())
        train_loss += loss_calc.item()
        loss_calc.backward()
        optimizer.step()
        correct += accuracy(result.detach().argmax(), label.argmax())
    return correct, train_loss

def data_dump():
    dataset = load_dataset("hyperpartisan_news_detection", "bypublisher")
    data = []
    for item in dataset['train']:
        data.append(item)
    with open("political_bias_data_train.json", 'w') as f:
        json.dump(data, f)


def main():
    # data_dump()
    dataset = utils.NewsDataset('political_bias_data_train.json', 200)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_)

    model = LSTM(input_size=embedding_size_,
                 output_size=dataset.output_size,
                 hidden_size=embedding_size_,
                 num_layers=1,
                 bidirectional=False,
                 num_epochs=1).to(torch_device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_)
    for epoch in range(200):
        if epoch % 10 == 0:
            correct_train, train_loss = train(model, loss, optimizer, dataloader)
            print("correct: {}".format(correct_train / dataset.__len__()))
            print("loss: {}".format(train_loss / dataset.__len__()))


if __name__ == "__main__":
    # testing
    main()