from network import BiLSTM_RNN
from dataloader import prepare_data
from utils import accuracy, train, evaluate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import torchtext
from torchtext import data
from torchtext import datasets

import time
import numpy as np
import argparse
import pickle

import os.path

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    REVIEW, SCORE, train_data, valid_data, test_data = prepare_data()

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = args.batch_size,
        device = device,
        sort_within_batch = True,
        sort_key=lambda x: len(x.review)
    )
    print('Finished loading data.')

    vocab_size = len(REVIEW.vocab)
    embedding_dim = 100 
    hidden_dim = args.hidden_dim
    output_dim = 5
    num_layers = args.num_layers
    dropout = args.dropout
    padding_index = REVIEW.vocab.stoi['<pad>']
    unknown_index = REVIEW.vocab.stoi['<unk>']

    model = BiLSTM_RNN(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, padding_index)
    
    #Load pretrained embeddings
    pretrained_embeddings = REVIEW.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    #Reset unknown and padding vectors
    model.embedding.weight.data[unknown_index] = torch.zeros(embedding_dim, device=device)
    model.embedding.weight.data[padding_index] = torch.zeros(embedding_dim, device=device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    epochs = args.num_epochs
    best_loss = np.inf

    if not (os.path.exists("./model")):
        os.mkdir("./model")

    for epoch in np.arange(epochs):

        start = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end = time.time()
        
        duration = time.strftime('%H:%M:%S', time.gmtime(end - start))

        if valid_loss < best_loss:
            best_loss = valid_loss
            with open('./model/model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        print(f'\nEpoch {epoch + 1} at {duration}')
        print(f'Train Loss: {train_loss:.3f} - Validation Loss: {valid_loss:.3f}')
        print(f'Train Acc: {train_acc:.2f} - Validation Acc: {valid_acc:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary') 
    parser.add_argument('--hidden_dim', type=int, default=100, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
