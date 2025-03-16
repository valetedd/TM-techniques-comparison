import argparse
import time
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import LogNormal, Dirichlet, Normal
from torch.distributions import kl_divergence
import pandas as pd
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from typing import Optional, List
import os
from tqdm import tqdm

from prodLDA_model import ProdLDA
import preprocessing as pp


# Arguments for CLI interface
parser = argparse.ArgumentParser(description='ProdLDA')

# Hidden layers
parser.add_argument('--hidden_size', type=int, default=1024,
                    help="number of hidden units for hidden layers")
# Number of topics
parser.add_argument('--num_topics', type=int, default=10,
                    help="number of topics")
# Dropout rate
parser.add_argument('--dropout', type=float, default=0.3,
                    help="dropout applied to layers (0 = no dropout)")
# data path
parser.add_argument('--data', type=str, default='data/UN_speeches/UNGDC_1946-2023.csv',
                    help="location of the data folder")
# lognormal or dirichlet
parser.add_argument('--use_lognormal', action='store_true',
                    help="Use LogNormal to approximate Dirichlet")
# max epochs
parser.add_argument('--epochs', type=int, default=20,
                    help="maximum training epochs")
# batch size
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help="batch size")
# LR
parser.add_argument('--lr', type=float, default=1e-2,
                    help="learning rate")
# Weight decay value
parser.add_argument('--wd', type=float, default=0,
                    help="weight decay used for regularization")
# KL anneal
parser.add_argument('--kl_anneal_epochs', type=int, default=10,
                    help="number of epochs over which to anneal KL weight from 0 to 1")
# Seed
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
# CUDA or not
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")

args = parser.parse_args() # Namespace containing all the hyperparams

torch.manual_seed(args.seed)


# Dataset class for the employed dataset

class UN_data(Dataset):
    def __init__(self, 
                 corpus : List[str] | List[List[str]] | List[List[tuple]], 
                 min_length : int,  
                 dct : Dictionary = None, 
                 preprocessed : bool = False):
        
        self.min_len = min_length
        if not preprocessed:
            dct_and_data = pp.LDA_pp(docs=corpus, from_preprocessed=False) 
            self.dct = dct_and_data[0]
            self.data = dct_and_data[1]
        else:
            if not dct:
                raise TypeError("If the data was already processed, a gensim Dictionary has to be passed to the 'dct' argument")
            self.dct = dct
            self.data = corpus
        print(f"Initialized dataset with:\n\t - {len(self.data)} documents;\n\t - A vocab of {len(self.dct)};\n\t - {self.num_tokens} tokens")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx : int):
        doc = self.data[idx]
        dense = torch.zeros(self.min_len)
        for id, count in doc:
            dense[id] = count
        return dense.to_sparse()
    
    @property
    def num_tokens(self):
        return sum(sum(count for _, count in doc) for doc in self.data)

### TRAINING IMPLEMENTATION ###

def standard_prior_like(posterior):

    if isinstance(posterior, LogNormal):
        loc = torch.zeros_like(posterior.loc)
        scale = torch.ones_like(posterior.scale)        
        prior = LogNormal(loc, scale)

    elif isinstance(posterior, Dirichlet):
        alphas = torch.ones_like(posterior.concentration)
        prior = Dirichlet(alphas)

    elif isinstance(posterior, Normal):
        prior_mu = torch.zeros_like(posterior.loc)
        prior_logvar = torch.zeros_like(posterior.scale)
        prior = posterior = Normal(prior_mu, (0.5 * prior_logvar).exp())

    return prior

def get_loss(inputs, outputs, posterior, device, beta=1.0):
    prior = standard_prior_like(posterior) # getting standard prior for KL
    # Getting the reconstruction loss (difference between the input-target and output-reconstruction)
    nll = - torch.sum(inputs * outputs).mean() # negative log-likelihood
    kld = beta * torch.sum(kl_divergence(posterior, prior).to(device)).mean() # Kullback-Liebler divergence
    return nll, kld


def evaluate(data_source, model, device, epoch):

    model.eval()
    total_nll = 0.0
    total_kld = 0.0
    total_words = 0
    
    for _, doc_batch in enumerate(data_source):
        test_data = doc_batch.to(device)
        test_pred, posterior = model(test_data)

        beta = min(1.0, epoch / args.kl_anneal_epochs) 
        nll, kld = get_loss(test_data, test_pred, posterior, device, beta=beta)

        total_nll += nll.item() 
        total_kld += kld.item() 
        total_words += test_data.sum().item()  # Count actual tokens in this batch

    # Normalize by the total number of tokens
    normalized_nll = total_nll / total_words
    normalized_kld = total_kld / total_words

    # Perplexity is directly from the normalized NLL
    ppl = math.exp(normalized_nll)
    
    return (normalized_nll, normalized_kld, ppl)


def train(data_source, model, optimizer, device, epoch):

    model.train()
    total_nll = 0.0
    total_kld = 0.0
    total_words = 0

    for _, doc_batch in enumerate(tqdm(data_source, desc=f"Epoch {epoch+1}/{args.epochs}")):
        data = doc_batch.to(device)
        total_words += data.sum().item()

        pred, posterior = model(data)
        beta = min(1.0, epoch / args.kl_anneal_epochs)
        nll, kld = get_loss(data, pred, posterior, device, beta=beta)

        total_nll += nll.item()
        total_kld += kld.item()

        optimizer.zero_grad()   
        loss = nll + kld
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    normalized_nll = total_nll / total_words
    normalized_kld = total_kld / total_words

    ppl = math.exp(normalized_nll) # perplexity
    return (normalized_nll, normalized_kld, ppl)


def get_savepath():
    dataset = args.data.rstrip('/').split('/')[-1]
    os.makedirs("./saves", exist_ok=True)
    path = './saves/hid{0:d}.tpc{1:d}{2}.{3}.pt'.format(
        args.hidden_size, args.num_topics,
        '.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
        dataset)
    return path


def print_top_words(beta, idx2word, n_words=10): # beta represents the topic-word distribution

    print("\n" + '-' * 30 + ' Topics ' + '-' * 30 + "\n")
    for i in range(len(beta)):
        line = ', '.join(
            [idx2word[j] for j in beta[i].argsort()[:-n_words-1:-1]])
        print(f"Topic {i+1}: {line};")


def main():
    
    print("Loading data")
    # data = pd.read_csv(args.data)["text"].to_numpy()[:50]
    bow_data, index = pp.load_pp("data/UN_PP", ("bow.pkl", "dictionary.dict"))
    bow_data = bow_data[:500]
    good_ids = set()
    for doc in bow_data:
        for idx, _ in doc:
            good_ids.add(idx)

    index.filter_tokens(good_ids=list(good_ids))

    if not index.id2token:
        index.id2token = {v: k for k, v in index.token2id.items()}
    
    
    # index, bow_data = preprocessing.LDA_pp(data, from_preprocessed=False)    
    train_split, test_split = train_test_split(bow_data,
                                             train_size=0.8,
                                             shuffle=True,
                                             random_state=42)
    vocab_size = len(index)
    
    train_data = UN_data(train_split, min_length=vocab_size, dct=index, preprocessed=True)
    test_data = UN_data(test_split, min_length=vocab_size, dct=index, preprocessed=True)

    train_dataloader = DataLoader(train_data, 
                                  batch_size=args.batch_size, 
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, 
                                 batch_size=args.batch_size, 
                                 shuffle=False)

    print("\tTraining data size: ", len(train_data))
    print("\tVocabulary size: ", vocab_size)
    print("Constructing model")
    print(args)

    device = torch.device('cpu' if args.nocuda else 'cuda')
    model = ProdLDA(
        vocab_size, 
        args.hidden_size, 
        args.num_topics,
        args.dropout, 
        # args.use_lognormal
        ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.wd)
    # Add to your code
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=0.3, 
                                                           patience=2,
                                                           min_lr=1e-5)

    best_loss = None

    print("\nStart training")
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_nll, train_kld, train_ppl = train(train_dataloader, model, optimizer, device, epoch)
            test_nll, test_kld, test_ppl = evaluate(test_dataloader, model, device, epoch)
            scheduler.step(test_nll + test_kld)
            
            print('-' * 80)
            meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time()-epoch_start_time)
            print(meta + "| train loss {:5.2f} ({:4.2f}) - Ratio: {:5.2f} "
                  "| train ppl {:5.2f}".format(
                      train_nll, train_kld, train_nll / train_kld, train_ppl))
            print(len(meta) * ' ' + "| test loss  {:5.2f} ({:4.2f}) "
                  "| test ppl  {:5.2f}".format(
                      test_nll, test_kld, test_ppl), flush=True)
            
            if best_loss is None or test_nll + test_kld < best_loss:
                best_loss = test_nll + test_kld
                with open(get_savepath(), 'wb') as f:
                    torch.save(model, f)
                
    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early')


    with open(get_savepath(), 'rb') as f:
        model = torch.load(f, weights_only=False)

    test_nll, test_kld, test_ppl = evaluate(test_dataloader, model, device, epoch)
    print('=' * 80)
    print("| End of training | test loss {:5.2f} ({:5.2f}) "
          "| test ppl {:5.2f}".format(
              test_nll, test_kld, test_ppl))
    print('=' * 80)
    beta = model.decoder.fc.weight.cpu().detach().numpy().T # topic-word distribution
    print(beta.shape, beta)
    id2token = index.id2token
    print_top_words(beta, id2token)

    
if __name__ == '__main__':
    main()