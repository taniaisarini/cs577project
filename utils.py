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

category_dict = {}

def category_to_num(category):
    return category_dict.get(category, 0)


class NewsDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        i = 0
        with open(filename) as f:
            num = 1
            for line in f:
                #  Start with few records
                if i >= 20:
                    break
                i += 1
                obj = json.loads(line)
                self.data.append(obj)
                if obj["category"] not in category_dict.keys():
                    category_dict[obj["category"]] = num
                    num += 1
        # self.num_categories = num - 1
        # TODO(me): remove this line
        self.num_categories = 42
        self.glove_embs = api.load("glove-wiki-gigaword-50")
        self.all_labels = []
        self.news_emb = []
        self.create_input()

    def __len__(self):
        return len(self.data)

    # TODO(me): make this a pre-processing step, its too slow.
    def create_input(self):
        for item in range(len(self.data)):
            print(item)
            # Get news paragraphs
            uf = urllib.request.urlopen(self.data[item]["link"])
            html = uf.read()
            soup = BeautifulSoup(html, features="html.parser")
            para = ''
            for data in soup.find_all("p"):
                para = para + (data.get_text())

            #  Convert to word embeddings
            x = torch.zeros(1, 50)
            for word in para.split(" "):
                try:
                    y = torch.tensor(None, self.glove_embs[word])
                except:
                    y = torch.randn(1, 50)
                x = torch.cat((x, y), 0)

            # Get the news category
            cat_num = category_to_num(self.data[item]["category"])
            labels = torch.zeros(self.num_categories)
            labels[cat_num] = 1
            self.all_labels.append(labels)
            self.news_emb.append(x)

    def __getitem__(self, item):
        return self.all_labels[item], self.news_emb[item]