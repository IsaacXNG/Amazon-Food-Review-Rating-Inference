# Amazon-Food-Review-Rating-Prediction
We attempt to predict the ratings of food reviews based on sentiment analysis using Bidirectional LSTMs with PyTorch.

Dataset download from https://www.kaggle.com/snap/amazon-fine-food-reviews/kernels

#### Dataset:
* Full dataset has 500,000+ reviews
* Balanced sample has 147500 reviews

#### Requirements:
* Python 3.7
* Standard Anaconda packages (e.g., numpy, pandas)
* PyTorch 1.0.0
* Torchtext 0.3.1
* Spacy 
* GloVe word embeddings: 6B 100d. https://nlp.stanford.edu/projects/glove/

#### Results:
* 70% accuracy on full dataset (60% on balanced test set)
* 0.69 F1 score
* 0.47 mean absolute error (in other words, on average, our predictions are only off by less than 0.5 stars)
* 0.36 seconds of training per epoch on one Nvidia Tesla K80. 5 - 10 epochs are required to reach convergence.

#### Samples: <br>
```
$ python predict.py "It was very fragrant and delicate. I definitely recommend." --print
Probability distribution:  [0.00227669 0.00171347 0.01063003 0.22481784 0.760562  ] 
5

$ python predict.py "I think it was enjoyable" --print
Probability distribution:  [0.05328422 0.11656603 0.20920141 0.4209052  0.20004311] 
4

$ python predict.py "It was slightly better than edible." --print
Probability distribution:  [0.1383879  0.23203094 0.30232716 0.23983285 0.08742111]  
3

$ python predict.py "It was disappointing." --print
Probability distribution:  [0.415574   0.4342737  0.14427686 0.00485574 0.00101966]  
2

$ python predict.py "It smelled foul and was very greasy." --print
Probability distribution:  [0.8074387  0.16070081 0.02730567 0.00314497 0.00140986] 
1

```
#### Usage: <br>
Once we download and extract Review.csv from Kaggle to the /data directory, we can use this command 
`$ python train.py`

And evaluate it on the test set using this command
`$ python eval.py`

train.py automatically calls dataloader.py which prepares the data for balancing and loading.

#### Try next: <br>
FastText word embeddings
