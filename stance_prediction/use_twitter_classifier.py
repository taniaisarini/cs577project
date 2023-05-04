from TwitterClassifier.neural_archs import DAN
import torch
import TwitterClassifier.utilsAdditional as utils_additional

dataset_train_path = 'cleaned_data_train.json'
dataset_val_path = 'cleaned_data_val.json'
dict_path = 'TwitterClassifier/total_vocab.json'
import utils

def main():
    # Each row of df -> input to DAN for item in same row of training data file
    train_df = utils_additional.file_to_dan(dataset_train_path, dict_path)
    val_df = utils_additional.file_to_dan(dataset_val_path, dict_path)
    op = utils_additional.dan_predict(train_df, 'TwitterClassifier/trump_dan.pth')
    print(len(op))





main()