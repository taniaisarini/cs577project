from torch.utils.data import Dataset
import torch
import json
from nltk.stem import *
import gensim.downloader as api
import spacy
import torch
import matplotlib.pyplot as plt
from nltk import word_tokenize

plt.style.use('ggplot')
from spacy import displacy

# hi
category_dict = {}

# make sure to run: spacy download en_core_web_lg
NER = spacy.load("en_core_web_lg")
categories = {}

def category_to_num(category):
    return category_dict.get(category, 0)


class NewsDataset(Dataset):
    def __init__(self, filename, perform_NER = True, use_twitter_classifier=False, DAN_df = None,
                 hyperpartisan = False):
        self.hyperpartisan = hyperpartisan
        self.DAN_df = DAN_df
        self.use_twitter_classifier = use_twitter_classifier
        self.perform_NER = perform_NER
        self.data = []
        with open(filename) as f:
            for line in f:
                obj = json.loads(line)
                for item in obj:
                    self.data.append(item)
        self.text = []
        self.preprocess()
        self.glove_embs = api.load("glove-wiki-gigaword-50")
        self.input_len = 150
        self.output_size = 2
        self.news_emb = []
        self.all_labels = []
        self.create_input()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.use_twitter_classifier:
            return self.DAN_df.iloc[[item]], self.all_labels[item]
        else:
            return self.news_emb[item], self.all_labels[item]

    def preprocess(self):
        max = 0
        for item in range(len(self.data)):
            # Get news paragraphs
            if self.perform_NER:
                para = self.distill(self.data[item]["text"])
            else:
                para = word_tokenize(self.data[item]["text"])
            self.text.append(para)
            if len(para) > max:
                max = len(para)
        self.input_len = max


    def create_input(self):
        for item in range(len(self.data)):
            # Get news paragraphs
            para = self.text[item]
            i = 0

            #  Convert to word embeddings
            x = torch.zeros(1, 50)
            for word in para:
                if i >= self.input_len:
                    break
                i += 1
                try:
                    y = torch.tensor(None, self.glove_embs[word])
                except:
                    y = torch.randn(1, 50)
                x = torch.cat((x, y), 0)
            if i < self.input_len:
                y = torch.zeros(self.input_len - i, 50)
                x = torch.cat((x, y), 0)

            self.news_emb.append(x)

            if (self.hyperpartisan):
                labels = torch.zeros(2)
                if self.data[item]['hyperpartisan']:
                    labels[0] = 1
                else:
                    labels[1] = 1
            else:
                labels = torch.zeros(self.output_size)
                if self.data[item]['bias'] == 0:
                    labels[0] = 1
                else:
                    labels[1] = 1

            self.all_labels.append(labels)

    def distill(self, text):
        # stop_words = set(stopwords.words('english'))
        # words = word_tokenize(text)
        # words = [w for w in words if not w.lower() in stop_words]
        return NER(text).ents


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least loss, then save the
    model state.
    credit: https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """

    def __init__(
            self, best_valid_loss=float('inf'), best_accuracy=-1
    ):
        self.best_valid_loss = best_valid_loss
        self.best_accuracy = best_accuracy

    def __call__(
            self, current_valid_loss, current_accuracy,
            epoch, model, optimizer, criterion, a_fn, l_fn
    ):
        if current_valid_loss < self.best_valid_loss:
            print("\nsaving for epoch: {} loss: {}\n".format(epoch, current_valid_loss))
            self.best_valid_loss = current_valid_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, l_fn)

        if current_accuracy > self.best_accuracy:
            print("\nsaving for epoch: {} accuracy: {}\n".format(epoch, current_accuracy))
            self.best_accuracy = current_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, a_fn)

def save_plots(train_acc, valid_acc, train_loss, valid_loss, a_fn, l_fn):
    """
    Function to save the loss and accuracy plots to disk.
    credit: https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(a_fn)

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(l_fn)