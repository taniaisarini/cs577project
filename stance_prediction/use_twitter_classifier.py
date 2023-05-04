from TwitterClassifier.neural_archs import DAN
import torch
import TwitterClassifier.utilsAdditional as utils_additional

dataset_train_path = 'cleaned_data_train.json'
dataset_val_path = 'cleaned_data_val.json'
dict_path = 'TwitterClassifier/total_vocab.json'
import utils

def call_twitter_classifier(data_path):
    # Each row of df -> input to DAN for item in same row of training data file
    df = utils_additional.file_to_dan(data_path, dict_path)

    op = utils_additional.dan_predict(df, 'TwitterClassifier/trump_dan.pth')
    return op