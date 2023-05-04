from torch.utils.data import Dataset

import numpy as np


def token_to_int(tokens, dict_key):
    index_list=[]
    for x in tokens:
        if x in dict_key.keys():
            index = dict_key[x]
        else:
            index = 0
        index_list.append(index)

    return index_list

class DAN_Dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = self.labels.iloc[idx, 0]

        sample = {'item': np.array(row), 'label': label}
        return sample

class DAN_Test_Dataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        sample = np.array(row)
        return sample


class RNN_Dataset(Dataset):
        def __init__(self, data, labels, seqLength):
            self.data = data
            self.labels = labels
            self.seqLength = seqLength

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]['intToken']
            label = self.labels.iloc[idx, 0]

            tempLen = len(row)
            if tempLen > self.seqLength:
                row = row[:self.seqLength]
            else:
                for i in range(self.seqLength - tempLen):
                    row.append(0)

            sample = {'item': np.array(row), 'label': label}

            return sample

