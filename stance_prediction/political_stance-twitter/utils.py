from torch.utils.data import Dataset
import torch
import re
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset
import torch
import json
from nltk.stem import *
import gensim.downloader as api
import spacy
import torch
import matplotlib.pyplot as plt
from nltk import word_tokenize

NER = spacy.load("en_core_web_lg")


class social_post_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.LongTensor(X)
        self.Y = torch.Tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    return text


def import_embeddings(path):
    emb_dict = {}
    num_words = 0
    arr_dim = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            x = line.split()
            emb_dict[x[0]] = np.asarray(x[1:], "float32")
            arr_dim = len(x[1:])
            num_words += 1
    keys_to_index = dict(zip(emb_dict.keys(), range(num_words)))
    glove_embs = np.empty((num_words, arr_dim))
    for i, items in enumerate(emb_dict.items()):
        glove_embs[i] = items[1]
    return keys_to_index, glove_embs


def import_data(path):
    df = pd.read_csv(path, usecols=[7, 20], encoding='ISO-8859-1')
    df.loc[df['bias'] == 'neutral', 'bias'] = 1
    df.loc[df['bias'] == 'partisan', 'bias'] = 0
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # turn the cleaned text into an array
    text = df['cleaned_text'].to_numpy()
    X = text[1:]

    # do the same for the y values (I had to use .values to get it to work for some reason)
    y = df['bias'].values
    y = y[1:]

    return X, y


class NewsDataset(Dataset):
    def __init__(self, filename, perform_NER=True, use_twitter_classifier=False, DAN_df=None,
                 hyperpartisan=False):
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
