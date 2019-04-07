import torch

def accuracy(preds, y):
    num_correct = torch.sum(torch.argmax(preds, dim=1, keepdim=False) == y)
    return num_correct.item()/len(y)

def train(model, iterator, optimizer, criterion):
    loss, acc = 0, 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()   
        preds = model(*batch.review)

        batch_loss = criterion(preds, batch.score)
        batch_loss.backward()
        optimizer.step()
        
        loss += batch_loss.item()
        acc += accuracy(preds, batch.score)
        
    return loss/len(iterator), acc/len(iterator)

def evaluate(model, iterator, criterion):
    loss, acc = 0, 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            preds = model(*batch.review)
            batch_loss = criterion(preds, batch.score)
            loss += batch_loss.item()
            acc += accuracy(preds, batch.score)
        
    return loss/len(iterator), acc/len(iterator)