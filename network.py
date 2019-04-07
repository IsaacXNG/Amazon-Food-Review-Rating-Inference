import torch
import torch.nn as nn

class BiLSTM_RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, padding_idx):
        super(BiLSTM_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, review, review_length):        
        embedded = self.dropout(self.embedding(review))
        padded = nn.utils.rnn.pack_padded_sequence(embedded, review_length)
        hidden = self.lstm(padded)[1][0]
        cat = torch.cat((hidden[-2], hidden[-1]), dim = 1)
        return self.fc(cat.squeeze(0))