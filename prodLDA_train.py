import argparse
import time
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions import LogNormal, Dirichlet
from torch.distributions import kl_divergence
import pandas as pd
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from typing import Optional, List
import os

from prodLDA_model import ProdLDA
from preprocessing import *

parser = argparse.ArgumentParser(description='ProdLDA')

# Hidden layers
parser.add_argument('--hidden_size', type=int, default=256,
                    help="number of hidden units for hidden layers")
# Number of topics
parser.add_argument('--num_topics', type=int, default=100,
                    help="number of topics")
# Dropout rate
parser.add_argument('--dropout', type=float, default=0.2,
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
parser.add_argument('--lr', type=float, default=5e-3,
                    help="learning rate")
# Weight decay value
parser.add_argument('--wd', type=float, default=0,
                    help="weight decay used for regularization")
# Iterations
parser.add_argument('--epoch_size', type=int, default=200,
                    help="number of training steps in an epoch")
# Seed
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
# CUDA or not
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
args = parser.parse_args()

torch.manual_seed(args.seed)

class UN_data(Dataset):
    def __init__(self, corpus : List[str] | List[List[str]], min_length : int,  dct : Dictionary = None, preprocessed : bool = False):
        self.min_len = min_length
        if not preprocessed:
            self._dct_and_data = LDA_pp(docs=corpus, from_preprocessed=preprocessed) 
            self.dct = self._dct_and_data[0]
            self.data = self._dct_and_data[1]
        else:
            if not dct:
                raise TypeError("If the data was already processed, a gensim Dictionary has to be passed to the 'dct' argument")
            self.dct = dct
            self.data = corpus
        print(f"Initialized dataset with {len(self.data)} tokens and a vocab of {len(self.dct)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx : int):
        doc = self.data[idx]
        dense = torch.zeros(self.min_len)
        for id, count in doc:
            dense[id] = count
        return dense


def recon_loss(targets, outputs): # Getting the reconstruction loss (difference between the input-target and output-reconstruction)
    nll = - torch.sum(targets * outputs) # Negative log-likelihood
    return nll


def standard_prior_like(posterior):
    if isinstance(posterior, LogNormal):
        loc = torch.zeros_like(posterior.loc)
        scale = torch.ones_like(posterior.scale)        
        prior = LogNormal(loc, scale)
    elif isinstance(posterior, Dirichlet):
        alphas = torch.ones_like(posterior.concentration)
        prior = Dirichlet(alphas)
    return prior
    

def get_loss(inputs, outputs, posterior, device):
    prior = standard_prior_like(posterior)
    nll = recon_loss(inputs, outputs)
    kld = torch.sum(kl_divergence(posterior, prior).to(device)) # Kullback-Liebler divergence
    return nll, kld # Two terms to be added to produce the loss fn


def evaluate(data_source, model, device):
    model.eval()
    total_nll = 0.0
    total_kld = 0.0
    total_words = 0
    size = len(data_source)
    for _, doc_batch in enumerate(data_source):
        test_data = doc_batch.to(device)
        test_pred, posterior = model(test_data)
        nll, kld = get_loss(test_data, test_pred, posterior, device)
        total_nll += nll.item() 
        total_kld += kld.item() 
        total_words += test_data.sum()

    total_kld, total_nll = total_kld / size, total_nll / size

    ppl = math.exp(total_nll * size / total_words) # Perplexity
    return (total_nll, total_kld, ppl)


def train(data_source, model, optimizer, device):

    model.train()
    total_nll = 0.0
    total_kld = 0.0
    total_words = 0
    size = args.batch_size

    for i in range(args.epoch_size):
        for doc_batch in data_source:
            data = doc_batch.to(device)
            pred, posterior = model(data)
            nll, kld = get_loss(data, pred, posterior, device)
            total_words += data.sum()

            optimizer.zero_grad()   
            loss = nll + kld
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_nll += nll.item()
            total_kld += kld.item()

    total_kld, total_nll = total_kld / size, total_nll / size

    ppl = math.exp(total_nll * size / total_words) # perplexity
    return (total_nll, total_kld, ppl)


def get_savepath(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    os.makedirs("./saves", exist_ok=True)
    path = './saves/hid{0:d}.tpc{1:d}{2}.{3}.pt'.format(
        args.hidden_size, args.num_topics,
        '.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
        dataset)
    return path


def print_top_words(beta, idx2word, n_words=10):
    print('-' * 30 + ' Topics ' + '-' * 30)
    for i in range(len(beta)):
        line = ' '.join(
            [idx2word[j] for j in beta[i].argsort()[:-n_words-1:-1]])
        print(line)


def main(args):
    
    print("Loading data")
    data = pd.read_csv(args.data)["text"].to_numpy()[:1000]

    index, bow_data = LDA_pp(data, from_preprocessed=False)    
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
        args.use_lognormal
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_loss = None

    print("\nStart training")
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_nll, train_kld, train_ppl = train(train_dataloader, model, optimizer, device)
            test_nll, test_kld, test_ppl = evaluate(test_dataloader, model, device)
            print('-' * 80)
            meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time()-epoch_start_time)
            print(meta + "| train loss {:5.2f} ({:4.2f}) "
                  "| train ppl {:5.2f}".format(
                      train_nll, train_kld, train_ppl))
            print(len(meta) * ' ' + "| test loss  {:5.2f} ({:4.2f}) "
                  "| test ppl  {:5.2f}".format(
                      test_nll, test_kld, test_ppl), flush=True)
            if best_loss is None or test_nll + test_kld < best_loss:
                best_loss = test_nll + test_kld
                with open(get_savepath(args), 'wb') as f:
                    torch.save(model, f)
                
    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early')


    with open(get_savepath(args), 'rb') as f:
        model = torch.load(f)

    test_nll, test_kld, test_ppl = evaluate(test_dataloader, model, device)
    print('=' * 80)
    print("| End of training | test loss {:5.2f} ({:5.2f}) "
          "| test ppl {:5.2f}".format(
              test_nll, test_kld, test_ppl))
    print('=' * 80)
    emb = model.decode.fc.weight.cpu().detach().numpy().T
    print_top_words(emb, index)

    
if __name__ == '__main__':
    main(args)