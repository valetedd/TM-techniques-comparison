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

print("CUDA:", torch.cuda.is_available())




class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Dropout(dropout)  # to avoid component collapse  
        )
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)
        

    def forward(self, inputs):
        h = self.enc(inputs)
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

    def guide(self, docs): # getting the approximate posterior and sampling theta
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
            count_param = self.decoder(theta) # Product of experts theta*beta (linear layer decoder weight)
            total_count = int(docs.sum(-1).max()) # getting max doc length as constant parameter for the multinomial
            pyro.sample(
                'obs',
                dist.Multinomial(total_count, count_param),
                obs=docs
            )

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T
    
    def train_model(
            self,
            docs,
            num_epochs : int = 50,
            learning_rate : float = 1e-2,
            optimizer = pyro.optim.Adam,
            device = "cpu",
            save : bool = False):
        
        # Training
        self.train()

        pyro.clear_param_store()

        opt_params = {"lr":learning_rate, "betas":(0.95, 0.999)}
        optimizer = torch.optim.AdamW

        sched_params = {"optimizer":optimizer, "optim_args":opt_params, "patience":5, "factor":0.2, "min_lr":1e-6}
        scheduler = pyro.optim.ReduceLROnPlateau(sched_params)

        svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())

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

        # saving model
        if save:
            import time

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            model_name = f"prodLDA_model_{current_time}.pt"
            optimizer_name = f"prodLDA_optimizer_{current_time}.pt"
            torch.save(self.state_dict(), f"data/results/prodLDA{model_name}")
            torch.save(scheduler.get_state(), f"data/results/prodLDA{optimizer_name}")
        return self
        

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
    
    def __getitem__(self, idx : int): # dense vector representation
        doc = self.data[idx]
        dense = torch.zeros(self.min_len)
        for id, count in doc:
            dense[id] = count
        return dense
    
    @property
    def num_tokens(self):
        return sum(sum(count for _, count in doc) for doc in self.data)


def get_dataloader(bow, vocab_size : int, batch_size : int, dct = None):
    # Handling data
    dataset = UN_data(
        corpus=bow, 
        min_length=vocab_size,
        dct=dct,
        preprocessed=True)

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        # num_workers=
        )
    return dataloader

# Checking results 

def get_topics(model, dct):
    print("\n" + '-' * 30 + ' Topics ' + '-' * 30 + "\n")
    beta = model.beta().numpy()

    topics = []

    idx2word = dct.id2token
    for i in range(len(beta)):
        topic = [idx2word[j] for j in beta[i].argsort()[:-10-1:-1]]
        if topic:
            print(f"Topic {i+1}: {topic};")
            topics.append(topic)

    # saving topics in txt
    str_topics = "\n".join([", ".join(topic) for topic in topics])
    with open("data/results/prodLDA.txt", mode="w", encoding="utf-8") as f:
        f.write(str_topics)
        return topics

def get_avg_coherence(topics, bow, dct):

    idx2word = dct.id2token

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
    return coherence




def main():
    dct, bow = pp.load_pp("data/UN_PP/", ("bow.pkl", "dictionary.dict"))  ### right order for linux
    if not isinstance(dct, Dictionary):
        bow, dct = dct, bow
    bow = bow[:500]

    good_ids = set() # getting IDs present in the bow slice
    for doc in bow:
        for idx, _ in doc:
            good_ids.add(idx)

    dct.filter_tokens(good_ids=list(good_ids)) # filtering vocabulary

    if not dct.id2token: # ensuring the id-token mapping is not empty due to bugs
        dct.id2token = {v: k for k, v in dct.token2id.items()}

    vocab_size = len(dct)

    # setting global variables
    SEED = 42
    torch.manual_seed(SEED)
    pyro.set_rng_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    num_topics = 20
    batch_size = 32
    learning_rate = 3e-3
    num_epochs = 150
    hidden = 256
    dropout = 0.1

    dataloader = get_dataloader(bow=bow, vocab_size=vocab_size, batch_size=batch_size, dct=dct)

    prodLDA = ProdLDA(
        vocab_size=vocab_size,
        num_topics=num_topics,
        hidden=hidden,
        dropout=dropout
    ).to(device)

    prodLDA.train_model(
        docs=dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
        save=False,
    )

    get_avg_coherence(get_topics(prodLDA, dct), bow, dct)


if __name__ == "__main__":
    main()