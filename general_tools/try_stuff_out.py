from datasets import load_dataset
import json
import csv
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer



category_dict = {}
num_hyperpartisan = {}

def data_dump():
    dataset = load_dataset("hyperpartisan_news_detection", "bypublisher")
    data = []

    for item in dataset['train']:
        bias = item['bias']
        if bias not in category_dict.keys():
            category_dict[bias] = 1
        else:
            category_dict[bias] += 1
        if item['hyperpartisan']:
            if bias not in num_hyperpartisan.keys():
                num_hyperpartisan[bias] = 1
            else:
                num_hyperpartisan[bias] += 1
    print(category_dict)
    print(num_hyperpartisan)




    # for item in dataset['train']:
    #     data.append(item)
    # with open("political_bias_data_train.json", 'w') as f:
    #     json.dump(data, f)
    # data = []
    # for item in dataset['validation']:
    #     data.append(item)
    # with open("political_bias_data_val.json", 'w') as f:
    #     json.dump(data, f)

# data_dump()

def nltk_sentiment():
    sa = SentimentAnalyzer()
    sia = SentimentIntensityAnalyzer()
    print(sia.polarity_scores(text="I really love harry potter"))
    print(sia.polarity_scores(text="I really hate 8 am finals"))

nltk_sentiment()
