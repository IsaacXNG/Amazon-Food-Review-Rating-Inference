import spacy
import torch

import pickle
import numpy as np
import argparse
import sys

def main(args):
    nlp = spacy.load('en')
    REVIEW, SCORE, model = None, None, None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.vocab_path, 'rb') as f:
        REVIEW = pickle.load(f)
    with open(args.label_path, 'rb') as f:
        SCORE = pickle.load(f)
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    model.to(device)
    print(predict(args.string, model, nlp.tokenizer, REVIEW, SCORE, device, args.print))

def predict(review, model, tokenizer, review_vocab, score_vocab, device, print_probs):
    tokens = [t.text for t in tokenizer(review)]
    indices = [review_vocab.stoi[t] for t in tokens]
    prediction = None

    with torch.no_grad():
        input = torch.LongTensor(indices).to(device).unsqueeze(1)
        likelihood = model(input, torch.tensor([len(tokens)]))
        probs = torch.softmax(likelihood, 0)
        pred_index = probs.argmax().item()
        scores = np.array(score_vocab.itos[0:5]).astype(int)
        prediction = score_vocab.itos[pred_index]

        if print_probs:
            print('Probability distribution: ', probs[np.argsort(scores)].cpu().numpy())
        
    return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('string', type=str, help='input review')
    parser.add_argument('--print', action='store_true', help='whether or not to print the probabilities')
    parser.add_argument('--model_path', type=str, default='./model/model.pkl', help='location of the model')
    parser.add_argument('--vocab_path', type=str, default='./vocab/vocab.pkl', help='location of vocabulary')
    parser.add_argument('--label_path', type=str, default='./vocab/labels.pkl', help='location of score dict')
    args = parser.parse_args()
    main(args)
