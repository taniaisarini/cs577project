from model import LSTM_Model
from utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if __name__ == '__main__':
    keys_to_index, glove_embs = import_embeddings('embedding/glove.6B.50d.txt')
    X, y = import_data('dataset/political_social_media.csv')
    num_words = len(keys_to_index)
    longest_sentence = 0
    for i, sentence in enumerate(X):
        longest_sentence = max(longest_sentence, len(sentence.split()))
    X_embs_index = np.full((X.shape[0], longest_sentence+1), num_words+1, dtype=np.int32)
    for i, sentence in enumerate(X):
        temp_list = []
        for word in sentence.split():
            temp_list.append(keys_to_index.get(word, num_words))
        X_embs_index[i, :len(temp_list)] = np.array(temp_list, np.int32)
    
    X_train, X_test, y_train, y_test = train_test_split(X_embs_index.astype(np.int32), y.astype(np.int32), test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    train_data = torch.utils.data.DataLoader(social_post_dataset(X_train, y_train), batch_size=32, shuffle=True)
    val_data = torch.utils.data.DataLoader(social_post_dataset(X_val, y_val), batch_size=32, shuffle=True)
    test_data = torch.utils.data.DataLoader(social_post_dataset(X_test, y_test), batch_size=32, shuffle=True)

    temp_embs = np.concatenate([glove_embs, np.random.random((1, glove_embs.shape[-1])), np.zeros((1, glove_embs.shape[-1]))])
    model = LSTM_Model(num_words+2, 50, pretrained=torch.Tensor(temp_embs))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_fn = torch.nn.BCELoss()
    train_loss = []
    val_loss = []
    best_acc = 0
    for epoch in tqdm(range(20)):
        model.train()
        losses = 0
        for X, y in train_data:
            optimizer.zero_grad()
            outputs = model(X).flatten()
            loss = loss_fn(outputs, y)
            losses += loss.item()
            loss.backward()
            optimizer.step()
            

        train_loss.append(losses / len(train_data))
        
        model.eval()
        losses = 0
        corr = 0
        with torch.no_grad():
            for X, y in val_data:
                outputs = model(X).flatten()
                loss = loss_fn(outputs, y)
                losses += loss.item()
                preds = np.where(outputs.numpy(), 1, 0)
                corr += sum(preds == y.numpy())
            acc = corr / len(test_data.dataset)
            val_loss.append(losses / len(val_data))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'model/lstm-pretrained.pt')
