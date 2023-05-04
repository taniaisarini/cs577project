import math
import numpy as np
import json
import pandas as pd
import re
import torch

from stance_prediction.TwitterClassifier.neural_archs import DAN
from stance_prediction.TwitterClassifier.utils import DAN_Test_Dataset

torch_device = torch.device("cpu")

def computeIdf(d, dict_freq):
    dict_idf = dict()
    for key in dict_freq.keys():
        dict_idf[key] = math.log((d + 0.0) / (dict_freq[key] + 1.0))
    return dict_idf


def tfidf(tokens_list, dict_idf):
    index = 0
    tfidf_df = pd.DataFrame(columns=dict_idf.keys())

    for tokens in tokens_list:
        # token freq for current tweet
        dict_curr = dict()
        for token in tokens:
            if token in dict_curr.keys():
                dict_curr[token] = dict_curr[token] + 1
            elif token in dict_idf.keys():
                dict_curr[token] = 1

        zeros = np.zeros(len(dict_idf))
        tfidf_df.loc[index] = zeros

        for key in dict_curr.keys():
            tfidf_df.at[index, key] = (dict_curr[key]/(len(dict_curr) + 0.0)) * dict_idf[key]

        index = index + 1

    return tfidf_df


def tokenization(text, tokens_re):
    return tokens_re.findall(text.lower())


def file_to_dan(dataset_path, dict_path):
    regex_str = [
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
        r'(?:[\w_]+)',  # other words
    ]
    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

    dict_freq = dict()

    with open(dict_path) as json_file:
        dict_index = json.load(json_file)

    for k in dict_index.keys():
        dict_freq[k] = 0

    tokens_list = []

    with open(dataset_path) as f:
        for line in f:
            obj = json.loads(line)
            for item in obj:
                text = item["text"]

                list_curr = []
                tokens = tokenization(text, tokens_re)

                tokens_list.append(tokens)

                for token in tokens:
                    if token in dict_index.keys():
                        if token not in list_curr:
                            dict_freq[token] = dict_freq[token] + 1
                            list_curr.append(token)

    dict_idf = computeIdf(len(tokens_list), dict_freq)

    tfidf_df = tfidf(tokens_list, dict_idf)

    return tfidf_df

def dan_predict(df, model_path):
    model = DAN(vocab_size=df.shape[1], hidden_dim=250, output_size=3)

    # model.load_state_dict(torch.load(model_path))
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError as e:
        print('Ignoring "' + str(e) + '"')
    model.eval()

    dataset = DAN_Test_Dataset(data=df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    test_pred = []

    for i, item in enumerate(dataloader):
        x = item.to(torch_device)
        x = x.to(torch.float32)

        output = model(x)

        test_pred = test_pred + output.argmax(1).tolist()

    return test_pred

