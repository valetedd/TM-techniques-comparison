import pyro
import pyro.distributions as dist
import pyro.optim
import torch
import torch.optim.optimizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
import preprocessing as pp
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from typing import List

print(torch.cuda.is_available())




class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            F.softplus(),
            nn.Linear(hidden, hidden),
            F.softplus(),
            nn.Dropout(dropout)  # to avoid component collapse  
        )
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)
        

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale


class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout)

    def guide(self, docs): # getting the approximate posterior using the encoder params
        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            logtheta_loc, logtheta_scale = self.encoder(docs)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            
    def model(self, docs): # equivalent of forward pass in PyTorch
        pyro.module("decoder", self.decoder)
        batch_size = docs.shape[0]
        with pyro.plate("documents", batch_size):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_loc = docs.new_zeros((batch_size, self.num_topics))
            logtheta_scale = docs.new_ones((batch_size, self.num_topics))
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1) # Topic distributions on the batch
            count_param = self.decoder(theta)
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs',
                dist.Multinomial(total_count, count_param),
                obs=docs
            )

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T
    

### TRAINING ###
    

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
        return dense
    
    @property
    def num_tokens(self):
        return sum(sum(count for _, count in doc) for doc in self.data)


# dct, bow = pp.load_pp("data/UN_PP", ("bow.pkl", "dictionary.dict"))  ### right order for linux
bow, dct = pp.load_pp("data/UN_PP", ("bow.pkl", "dictionary.dict"))
bow = bow[:500]
good_ids = set()
for doc in bow:
    for idx, _ in doc:
        good_ids.add(idx)

dct.filter_tokens(good_ids=list(good_ids))

if not dct.id2token:
    dct.id2token = {v: k for k, v in dct.token2id.items()}

vocab_size = len(dct)

# setting global variables
seed = 42
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
num_topics = 20
batch_size = 32
learning_rate = 5e-3
num_epochs = 150

# Handling data
dataset = UN_data(
    corpus=bow, 
    min_length=vocab_size,
    dct=dct,
    preprocessed=True)

docs = DataLoader(
    dataset=dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    # num_workers=8
    )

# Training
pyro.clear_param_store()

prodLDA = ProdLDA(
    vocab_size=vocab_size,
    num_topics=num_topics,
    hidden=512,
    dropout=0.2
)
prodLDA.to(device)

opt_params = {"lr":learning_rate, "betas":(0.95, 0.999)}
optimizer = torch.optim.AdamW

sched_params = {"optimizer":optimizer, "optim_args":opt_params, "patience":5, "factor":0.2, "min_lr":1e-6}
scheduler = pyro.optim.ReduceLROnPlateau(sched_params)

svi = SVI(prodLDA.model, prodLDA.guide, scheduler, loss=TraceMeanField_ELBO())

bar = trange(num_epochs)
try:
    for epoch in bar:
        running_loss = 0.0
        for doc in docs:
            batch_docs = doc.to(device)
            loss = svi.step(batch_docs)
            running_loss += loss / batch_docs.size(0)
        scheduler.step(metrics=loss)
        epoch_loss = running_loss / len(docs)
        bar.set_postfix(epoch_loss='{:.2f}'.format(epoch_loss))
except KeyboardInterrupt:
    print("Exiting training early")


# Checking results 
print("\n" + '-' * 30 + ' Topics ' + '-' * 30 + "\n")
beta = prodLDA.beta().numpy()

topics = []
idx2word = dct.id2token
for i in range(len(beta)):
    topic = [idx2word[j] for j in beta[i].argsort()[:-10-1:-1]]
    if topic:
        print(f"Topic {i+1}: {topic};")
        topics.append(topic)

texts = [[idx2word[id] for id, _ in doc] for doc in bow]

print("Getting coherence scores...")

coherence_model = CoherenceModel(
    topics=topics,
    texts=texts,
    dictionary=dct,
    coherence='c_v'
)
    
coherence = coherence_model.get_coherence()
print(f"Coherence: {coherence}")

str_topics = [", ".join(topic)+"\n" for topic in topics]
with open("data/results/prodLDA.txt", mode="w", encoding="utf-8") as f:
    f.write()

def main():
    pass

if __name__ == "__main__":
    main()