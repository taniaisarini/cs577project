import numpy as np
from torch.utils.data import Dataset
import torch
import json
import urllib
import csv
import pandas as pd
from nltk.stem import *
from nltk.wsd import lesk
import gensim.downloader as api
from nltk.corpus import wordnet as wn
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

category_dict = {}

def category_to_num(category):
    return category_dict.get(category, 0)


class NewsDataset(Dataset):
    def __init__(self, filename, num_items = 1000):
        self.data = []
        self.num_items = num_items
        with open(filename) as f:
            for line in f:
                obj = json.loads(line)
                for item in obj:
                    if (len(self.data) >= self.num_items):
                        break
                    self.data.append(item)
        self.text = []
        self.preprocess()
        self.glove_embs = api.load("glove-wiki-gigaword-50")
        self.input_len = 150
        self.output_size = 5
        self.news_emb = []
        self.all_labels = []
        self.create_input()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.news_emb[item], self.all_labels[item]

    def preprocess(self):
        max = 0
        for item in range(len(self.data)):
            # Get news paragraphs
            para = self.remove_stop_words(self.data[item]["text"])
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

            labels = torch.zeros(self.output_size)
            labels[self.data[item]["bias"]] = 1
            self.all_labels.append(labels)

    def remove_stop_words(self, text):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        words = [w for w in words if not w.lower() in stop_words]
        return words