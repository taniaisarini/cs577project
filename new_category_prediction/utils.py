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
import spacy

category_dict = {}
NER = spacy.load("en_core_web_lg")

def category_to_num(category):
    return category_dict.get(category, 0)


class NewsDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        with open(filename) as f:
            num = 0
            for line in f:
                obj = json.loads(line)
                self.data.append(obj)
                if obj["category"] not in category_dict.keys():
                    category_dict[obj["category"]] = num
                    num += 1
        self.num_categories = num
        self.input_len = 0
        self.glove_embs = api.load("glove-wiki-gigaword-50")
        self.stop_words = set(stopwords.words('english'))
        self.text = []
        self.preprocess()
        self.all_labels = []
        self.news_emb = []
        self.create_input()

    def __len__(self):
        return len(self.data)

    # TODO(me): make this a pre-processing step, its too slow.
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

            # Get the news category
            cat_num = category_to_num(self.data[item]["category"])
            labels = torch.zeros(self.num_categories)
            labels[cat_num] = 1
            self.all_labels.append(labels)
            self.news_emb.append(x)

    def preprocess(self):
        max = 0
        for item in range(len(self.data)):
            # Get news paragraphs
            para = self.distill(self.data[item]["text"])
            self.text.append(para)
            if len(para) > max:
                max = len(para)
        self.input_len = max

    def distill(self, text):
        # words = word_tokenize(text)
        # words = [w for w in words if not w.lower() in self.stop_words]
        # return words
        return NER(text).ents

    def __getitem__(self, item):
        return self.all_labels[item], self.news_emb[item]