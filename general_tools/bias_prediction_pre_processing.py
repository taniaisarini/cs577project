from datasets import load_dataset
import json

train_file = "../stance_prediction/political_bias_data_train.json"
val_file = "../stance_prediction/political_bias_data_val.json"
def create_balanced_dataset():
    data = []
    num_items = 500
    per_cat = {'L': 0, 'R': 0}
    with open(val_file) as f:
        for line in f:
            obj = json.loads(line)
            for item in obj:
                if (len(data) >= num_items):
                    break
                if item['hyperpartisan']:
                    if ((item['bias'] == 0) and per_cat['L'] < num_items/2):
                        per_cat['L'] += 1
                        data.append(item)
                    elif ((item['bias'] == 4) and per_cat['R'] < num_items/2):
                        per_cat['R'] += 1
                        data.append(item)
    with open("../stance_prediction/cleaned_data_val.json", 'w') as f:
        json.dump(data, f)

create_balanced_dataset()

