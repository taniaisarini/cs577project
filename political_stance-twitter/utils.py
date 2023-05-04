from torch.utils.data import Dataset
import torch
import re
import numpy as np
import pandas as pd

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

    #turn the cleaned text into an array
    text = df['cleaned_text'].to_numpy()
    X = text[1:]

    #do the same for the y values (I had to use .values to get it to work for some reason)
    y = df['bias'].values
    y = y[1:]

    return X, y