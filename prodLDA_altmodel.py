import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, inputs):
        h = F.elu(self.bn1(self.fc1(inputs)))
        h = F.elu(self.bn2(self.fc2(h)))
        return self.drop(h)

class TopicEncoder(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_size, num_topics)
        self.fc_logvar = nn.Linear(hidden_size, num_topics)
        
    def forward(self, hidden):
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.fc = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)
        
        # Initialize weights with Xavier normal
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        return F.log_softmax(self.fc(inputs), dim=1)

class ProdLDA(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics, dropout):
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_size, dropout)
        self.topic_encoder = TopicEncoder(hidden_size, num_topics)
        self.decoder = Decoder(vocab_size, num_topics, dropout)
        
        # Prior parameters: standard normal
        self.prior_mu = nn.Parameter(torch.zeros(num_topics), requires_grad=False)
        self.prior_logvar = nn.Parameter(torch.zeros(num_topics), requires_grad=False)

    def forward(self, inputs):
        # Encode input
        hidden = self.encoder(inputs)
        mu, logvar = self.topic_encoder(hidden)
        
        # Create posterior distribution
        posterior = Normal(mu, (0.5 * logvar).exp())
        
        # Sample from posterior
        if self.training:
            z = posterior.rsample()
        else:
            z = posterior.mean
        
        # Convert to topic proportions
        theta = F.softmax(z, dim=1)
        
        # Decode
        recon = self.decoder(theta)
        
        # # Compute KL divergence
        # prior = Normal(self.prior_mu, (0.5 * self.prior_logvar).exp())
        # kl = kl_divergence(posterior, prior).mean(dim=0).sum()
        
        return recon, posterior