import pandas as pd
import re
import os.path

import torch

import torchtext
from torchtext import data
from torchtext import datasets

import pickle

def prepare_data(vocab_size = 20000, train=True):
    if not (os.path.isfile("./data/amazon_train.csv") and os.path.isfile("./data/amazon_test.csv")):
        generate_samples()
    
    SCORE = data.LabelField(dtype=torch.long)
    REVIEW = data.Field(tokenize='spacy', lower=True, include_lengths=True)
    print('Finished loading spacy.')

    fields = [('score', SCORE), ('review', REVIEW)]

    train_path = "amazon_train.csv"
    test_path = "amazon_test.csv"
    train_data, valid_data, test_data = None, None, None

    if(train):
        test_path = None
    else:
        train_path = None

    data_csv = data.TabularDataset.splits(
                                        path = './data',
                                        train = train_path,
                                        test = test_path,
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True
    )

    if(train):
        train_data, valid_data = data_csv[0].split(split_ratio=0.8)
    else:
        test_data = data_csv[0]

    if not (os.path.exists("./vocab")):
        os.mkdir("./vocab")
    if not (os.path.isfile("./vocab/vocab.pkl") and os.path.isfile("./vocab/labels.pkl")):
        REVIEW.build_vocab(train_data, max_size = vocab_size, vectors="glove.6B.100d", unk_init = torch.Tensor.normal_)
        SCORE.build_vocab(train_data)
        print("Finished building vocabulary.")
        with open("./vocab/vocab.pkl", 'wb') as f:
            pickle.dump(REVIEW.vocab, f)
        with open("./vocab/labels.pkl", 'wb') as f:
            pickle.dump(SCORE.vocab, f)
    else:
        with open("./vocab/vocab.pkl", 'rb') as f:
            REVIEW.vocab = pickle.load(f)
        with open("./vocab/labels.pkl", 'rb') as f:
            SCORE.vocab = pickle.load(f)
    
    return REVIEW, SCORE, train_data, valid_data, test_data

def generate_samples():
    reviewdf = pd.read_csv("./data/Reviews.csv", usecols=["Score","Text"])

    #Make all review scores equal distribution by undersampling
    balanced = None
    for i in reviewdf["Score"].unique():
        balanced = pd.concat([balanced, reviewdf[reviewdf["Score"] == i][0:29500]])
    balanced = balanced.sample(frac=1).reset_index(drop=True)

    def clean_review(x):
        x = re.sub('\\\\|<[^>]+>', '', x) #remove <br>
        x = re.sub(r'\([^)]*\)', '', x) #remove (in-between parenthesis)
        x = x.replace('"',"")
        x = x.replace("'","")
        return x

    balanced["Text"] = balanced["Text"].apply(lambda x: clean_review(x))
    balanced[0:29500*4].to_csv('./data/amazon_train.csv', index=False)
    balanced[29500*4:].to_csv('./data/amazon_test.csv', index=False)
    print('Generated training and test files.')

