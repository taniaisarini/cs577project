from TwitterClassifier.neural_archs import DAN
import torch
import TwitterClassifier.utilsAdditional as utils_additional

dataset_train_path = 'cleaned_data_train.json'
dataset_val_path = 'cleaned_data_val.json'
dict_path = 'TwitterClassifier/total_vocab.json'
def main():
    # Each row of df -> input to DAN for item in same row of training data file
    train_df = utils_additional.file_to_dan(dataset_train_path, dict_path)
    # ith row should be equal to ith text
    print((train_df.iloc[[7]]))
    train_x = torch.zeros(10,10)
    model = DAN(vocab_size=train_x.shape[1], hidden_dim=250, output_size=3)

main()