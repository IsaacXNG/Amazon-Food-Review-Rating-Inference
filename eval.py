from network import BiLSTM_RNN
from dataloader import prepare_data
from utils import evaluate

import torch
import torch.nn as nn
from torchtext import data

import argparse
import os.path
import pickle

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    REVIEW, SCORE, train_data, valid_data, test_data = prepare_data(train=False)

    test_iterator = data.Iterator(
        test_data, 
        batch_size = args.batch_size, 
        device = device, 
        sort_within_batch = True, 
        sort_key=lambda x: len(x.review)
    )
    print('Finished loading data.')

    model = None
    with open(args.model_path,'rb') as f:
        model = pickle.load(f)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    loss, acc = evaluate(model, test_iterator, criterion)
    print('Testing data:')
    print(f'Loss: {loss:.3f}')
    print(f'Acc: {acc:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/model.pkl', help='location of the model')
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    main(args)