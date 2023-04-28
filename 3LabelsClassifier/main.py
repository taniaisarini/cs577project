import gensim
import math
import numpy as np
import os.path
import pandas as pd
import re
import torch

from neural_archs import DAN, RNN
from utils import DAN_Dataset, token_to_int, RNN_Dataset

torch.set_default_tensor_type(torch.FloatTensor)
torch_device = torch.device("cpu")

def tokenization(text, tokens_re):
    return tokens_re.findall(text.lower())


def buildDictionary(df):
    dict_freq = dict()
    dict_index = dict()
    vocab_count = 1
    for x in df['tokens']:
        list_curr = []
        for token in x:
            if token in dict_freq.keys():
                if token not in list_curr:
                    dict_freq[token] = dict_freq[token] + 1
                    list_curr.append(token)
            else:
                dict_freq[token] = 1
                dict_index[token] = vocab_count
                vocab_count = vocab_count + 1
                list_curr.append(token)

    return dict_freq, dict_index

def computeIdf(d, dict_freq):
    dict_idf = dict()
    for key in dict_freq.keys():
        dict_idf[key] = math.log((d + 0.0) / (dict_freq[key] + 1.0))
    return dict_idf

def tfidf(df, dict_idf):
    index = 0
    tfidf_df = pd.DataFrame(columns=dict_idf.keys())

    for tokens in df['tokens']:
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

def label_processing(df):
    label_df = pd.DataFrame()

    dict_sentiment = {'NONE': 0,
                     'AGAINST': 1,
                     'FAVOR': 2
                     }

    label_df['label'] = [dict_sentiment[x] for x in df['label']]

    return label_df

def DAN_train(train_x, train_y, epoch, lr):
    full_dataset = DAN_Dataset(data=train_x, labels=train_y)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = DAN(vocab_size=train_x.shape[1], hidden_dim=250, output_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accuracy_list = []
    val_accuracy_list = []
    best_val_accuracy = 0
    best_model_state = None

    for e in range(epoch):
        print("epoch: "+str(e))
        # train
        model.train()
        correct = 0
        total_l = 0
        for i, item in enumerate(train_dataloader):
            x = item['item'].to(torch_device)
            x = x.to(torch.float32)
            y = item['label'].to(torch_device)

            optimizer.zero_grad()

            output = model(x)

            loss = torch.nn.functional.cross_entropy(output, y)
            total_l = total_l + loss

            correct += (output.argmax(1) == y).sum()

            loss.backward()
            optimizer.step()

        train_accuracy = (correct/train_size).item()
        train_accuracy_list.append(train_accuracy)

        # validation
        model.eval()
        correct = 0
        total_l = 0
        for i, item in enumerate(val_dataloader):
            x = item['item'].to(torch_device)
            x = x.to(torch.float32)
            y = item['label'].to(torch_device)

            output = model(x)

            loss = torch.nn.functional.cross_entropy(output, y)
            total_l = total_l + loss

            correct += (output.argmax(1) == y).sum()
        val_accuracy = (correct / val_size).item()
        if best_val_accuracy < val_accuracy:
            print("updated: " + str(best_val_accuracy) + " -> " +str(val_accuracy) )
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

        val_accuracy_list.append(val_accuracy)

    print(train_accuracy_list)
    print(val_accuracy_list)

    return best_model_state


def RNN_train(train_x, train_y, epoch, lr, seqLength, vocab_size):
    full_dataset = RNN_Dataset(data=train_x, labels=train_y, seqLength=seqLength)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = RNN(vocab_size=vocab_size, hidden_dim=128, n_layer=1, bidirectional=True, output_size=3, seqLength=seqLength)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accuracy_list = []
    val_accuracy_list = []
    best_val_accuracy = 0
    best_model_state = None

    for e in range(epoch):
        print("epoch: "+str(e))
        # train
        model.train()
        correct = 0
        total_l = 0
        for i, item in enumerate(train_dataloader):
            x = item['item'].to(torch_device)
            x = x.to(torch.long)
            y = item['label'].to(torch_device)

            optimizer.zero_grad()

            output = model(x)

            loss = torch.nn.functional.cross_entropy(output, y)
            total_l = total_l + loss

            correct += (output.argmax(1) == y).sum()

            loss.backward()
            optimizer.step()

        train_accuracy = (correct/train_size).item()
        train_accuracy_list.append(train_accuracy)

        # validation
        model.eval()
        correct = 0
        total_l = 0
        for i, item in enumerate(val_dataloader):
            x = item['item'].to(torch_device)
            x = x.to(torch.long)
            y = item['label'].to(torch_device)

            output = model(x)

            loss = torch.nn.functional.cross_entropy(output, y)
            total_l = total_l + loss

            correct += (output.argmax(1) == y).sum()
        val_accuracy = (correct / val_size).item()
        if best_val_accuracy < val_accuracy:
            print("updated: " + str(best_val_accuracy) + " -> " +str(val_accuracy) )
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

        val_accuracy_list.append(val_accuracy)

    print(train_accuracy_list)
    print(val_accuracy_list)

    return best_model_state

def int_tokenization(df, dict_index):
    int_df = pd.DataFrame()

    int_df['intToken'] = df['tokens'].apply(lambda x: token_to_int(x, dict_index))

    return int_df


def DataPreprocessing(clean_train_df, clean_test_df):
    print("DataPreprocessing")

    train_df = clean_train_df.copy()
    test_df = clean_test_df.copy()

    regex_str = [
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
        r'(?:[\w_]+)',  # other words
    ]
    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

    train_df['tokens'] = train_df['text'].apply(lambda x: tokenization(x, tokens_re))
    test_df['tokens'] = test_df['text'].apply(lambda x: tokenization(x, tokens_re))

    dict_freq, dict_index = buildDictionary(train_df)

    # train data

    int_df = int_tokenization(train_df, dict_index)

    dict_idf = computeIdf(len(train_df['label']), dict_freq)

    tfidf_df = tfidf(train_df, dict_idf)
    label_train_df = label_processing(train_df)

    # test data

    int_test_df = int_tokenization(test_df, dict_index)

    tfidf_test_df = tfidf(test_df, dict_idf)
    label_test_df = label_processing(train_df)

    return tfidf_df, int_df, label_train_df, tfidf_test_df, int_test_df, label_test_df, len(dict_index)


if __name__ == '__main__':
    dict_sentiment = {'NONE': 0,
                     'AGAINST': 1,
                     'FAVOR': 2
                     }
    sentiment_list = ['NONE', 'AGAINST', 'FAVOR']

    biden_train_file = 'kawintiranon-stance-detection/biden_stance_train_public.csv'
    biden_test_file = 'kawintiranon-stance-detection/biden_stance_test_public.csv'

    trump_train_file = 'kawintiranon-stance-detection/trump_stance_train_public.csv'
    trump_test_file = 'kawintiranon-stance-detection/trump_stance_test_public.csv'

    train_df = pd.read_csv(biden_train_file)
    test_df = pd.read_csv(biden_test_file)


    tfidf_df, int_df, label_train_df, tfidf_test_df, int_test_df, label_test_df, vocab_size = DataPreprocessing(train_df, test_df)


    meanLength = int(int_df['intToken'].str.len().mean())

    # if not os.path.isfile('biden_train_tfidf.csv'):
    #     tfidf_df, int_df, label_train_df, tfidf_test_df, int_test_df, label_test_df = DataPreprocessing(train_df, test_df)
    #
    #     tfidf_df.to_csv('biden_train_tfidf.csv')
    #     int_df.to_csv('biden_train_int.csv')
    #     label_train_df.to_csv('biden_train_label.csv')
    #     tfidf_test_df.to_csv('biden_test_tfidf.csv')
    #     int_test_df.to_csv('biden_test_int.csv')
    #     label_test_df.to_csv('biden_test_label.csv')
    # else:
    #     tfidf_df = pd.read_csv('biden_train_tfidf.csv', index_col=0)
    #     int_df = pd.read_csv('biden_train_int.csv', index_col=0)
    #     temp_df = []
    #     for x in int_df['intToken']:
    #         temp = []
    #         for s in x[1:-1].split(','):
    #             temp.append(int(s))
    #         temp_df.append(temp)
    #     int_df['intToken'] = temp_df
    #     label_train_df = pd.read_csv('biden_train_label.csv', index_col=0)
    #
    #     tfidf_test_df = pd.read_csv('biden_test_tfidf.csv', index_col=0)
    #     int_test_df = pd.read_csv('biden_test_int.csv', index_col=0)
    #     temp_df = []
    #     for x in int_test_df['intToken']:
    #         temp = []
    #         for s in x[1:-1].split(','):
    #             temp.append(int(s))
    #         temp_df.append(temp)
    #     int_test_df['intToken'] = temp_df
    #     label_test_df = pd.read_csv('biden_test_label.csv', index_col=0)

    train_tfidf_x = tfidf_df
    train_int_x = int_df
    train_y = label_train_df

    test_tfidf_x = tfidf_test_df
    test_int_x = int_test_df
    test_y = label_test_df

    # dan
    # dan_best_model = DAN_train(train_tfidf_x, train_y, 10, 0.001)

    print(meanLength)

    # rnn
    rnn_best_model = RNN_train(train_int_x, train_y, 100, 0.0001, seqLength=meanLength, vocab_size=vocab_size)



