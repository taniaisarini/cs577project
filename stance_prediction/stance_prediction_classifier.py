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

embedding_size_ = 50
learning_rate_ = 0.0001
batch_size_ = 20
weight_decay_=1e-5

a_fn_plt = 'output/NER/accuracy_decay_e-03.png'
l_fn_plt = 'output/NER/loss_decay_e-03.png'
a_fn_model = 'output/NER/best_model_accuracy_decay_e-03.pth'
l_fn_model = 'output/NER/best_model_loss_decay_e-03.pth'

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
    for x, label in dataloader:
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
    val_loss = 0
    correct = 0
    for x, label in dataloader:
        output = model.forward(x.detach())
        # Using torchhe output of last element in sequence as predicted value
        result = output[:, -1:, :].to(torch.float).squeeze(1)
        loss_calc = loss(result, label)
        val_loss += loss_calc.item()
        correct += accuracy(result.detach().argmax(dim=1), label.argmax(dim=1))
    return correct, val_loss


def data_dump():
    dataset = load_dataset("hyperpartisan_news_detection", "bypublisher")
    data = []
    for item in dataset['train']:
        data.append(item)
    with open("political_bias_data_train.json", 'w') as f:
        json.dump(data, f)
    data = []
    for item in dataset['validation']:
        data.append(item)
    with open("political_bias_data_val.json", 'w') as f:
        json.dump(data, f)


def main():
    # data_dump()
    dataset_train = utils.NewsDataset('cleaned_data_train.json', perform_NER=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_,
                                                   shuffle=True)
    dataset_val = utils.NewsDataset('cleaned_data_val.json', perform_NER=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size_,
                                                 shuffle=True)

    model = LSTM(input_size=embedding_size_,
                 output_size=dataset_train.output_size,
                 hidden_size=embedding_size_,
                 num_layers=5,
                 bidirectional=False,
                 num_epochs=1).to(torch_device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_,
                                 weight_decay=weight_decay_)
    save_best_model = utils.SaveBestModel()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(200):
        correct_train_raw, train_loss_raw = train(model, loss, optimizer, dataloader_train)
        correct_val_raw, val_loss_raw = validate(model, loss, dataloader_val)

        # Get losses and accuracies and save them
        correct_train = correct_train_raw / dataset_train.__len__()
        train_loss = train_loss_raw / dataset_train.__len__()
        correct_val = correct_val_raw / dataset_val.__len__()
        val_loss = val_loss_raw / dataset_val.__len__()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(correct_train)
        val_accuracies.append(correct_val)

        if epoch % 20 == 0:
            print("******* epoch {} *******".format(epoch))
            print("training accuracy: {}".format(correct_train))
            print("training loss: {}".format(train_loss))
            print("validation accuracy: {}".format(correct_val))
            print("validation loss: {}".format(val_loss))
        save_best_model(val_loss,
                        correct_val,
                        epoch,
                        model,
                        optimizer,
                        loss,
                        a_fn_model,
                        l_fn_model)
    utils.save_plots(train_accuracies, val_accuracies, train_losses, val_losses,
                     a_fn_plt, l_fn_plt)
    print("highest train accuracy: {}".format(max(train_accuracies)))
    print("highest val accuracy: {}".format(max(val_accuracies)))

if __name__ == "__main__":
    # run data dump if running for the first time.
    # data_dump()

    main()