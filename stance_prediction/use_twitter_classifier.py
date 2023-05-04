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
    # dataset_train = utils.NewsDataset('cleaned_data_train.json', use_twitter_classifier=True, train_df=train_df)
    # dataset_train = utils.NewsDataset('cleaned_data_train.json', use_twitter_classifier=True, train_df=val_df)
    # ith row should be equal to ith text
    train_x = train_df.iloc[[3]]
    model = DAN(vocab_size=train_x.shape[1], hidden_dim=250, output_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint = torch.load('TwitterClassifier/dan_best_model.pth')

    # Getting key error for line 25 - 28 fields,
    # following https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    model.eval()

    output = []
    for row in train_df.iterrows():
        output.append(model(row))





main()