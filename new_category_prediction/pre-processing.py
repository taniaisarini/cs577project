import numpy as np
from torch.utils.data import Dataset
import torch
import json
import urllib
import csv
import pandas as pd
import os
from nltk.stem import *
from nltk.wsd import lesk
import gensim.downloader as api
from nltk.corpus import wordnet as wn
import nltk
from bs4 import BeautifulSoup

top_categories = {
    'POLITICS': 0,
    'WELLNESS': 0,
'ENTERTAINMENT': 0,
# 'STYLE & BEAUTY': 0,
# 'TRAVEL': 0,
# 'PARENTING': 0,
# 'HEALTHY LIVING': 0,
'QUEER VOICES': 0,
'FOOD & DRINK': 0,
'BUSINESS': 0,
# 'COMEDY': 0,
'SPORTS': 0,
'BLACK VOICES': 0,
# 'HOME & LIVING': 0,
# 'PARENTS': 0,
}


def append_record(record):
    with open('category_data.json', 'a') as f:
        json.dump(record, f)
        f.write(os.linesep)


filename = 'News_Category_Dataset_v3.json'
with open(filename) as f, open('category_data.json', 'a') as wf:
    total = 0
    for line in f:
        obj = json.loads(line)
        new_obj = {}
        if obj["category"] in top_categories.keys():
            if top_categories[obj["category"]] < 100:
                if total > len(top_categories) * 100:
                    break
                total += 1
                top_categories[obj["category"]] += 1
                new_obj["category"] = obj["category"]
                uf = urllib.request.urlopen(obj["link"])
                html = uf.read()
                soup = BeautifulSoup(html, features="html.parser")
                para = ''
                for data in soup.find_all("p"):
                    para = para + (data.get_text())
                new_obj["text"] = para
                append_record(new_obj)
