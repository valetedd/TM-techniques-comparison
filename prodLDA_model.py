import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal, Dirichlet


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_size)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        h1 = self.bn1(F.softplus(self.fc1(inputs)))
        h2 = self.bn2(F.softplus(self.fc2(h1)))
        h3 = self.bn3(F.softplus(self.fc3(h2)))
        return self.drop(h3)


class HiddenToLogNormal(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)
        
        self.fc_sigma = nn.Linear(hidden_size, num_topics)
        self.bn_sigma = nn.BatchNorm1d(num_topics)

    def forward(self, hidden):
        mu = self.bnmu(self.fcmu(hidden)) 
        log_var = self.bn_sigma(self.fc_sigma(hidden))
        dist = LogNormal(mu, (0.5 * log_var).exp())
        return dist

        
class HiddenToDirichlet(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_topics)
        self.bn = nn.BatchNorm1d(num_topics)

    def forward(self, hidden):
        alphas = self.bn(self.fc(hidden)).exp() # ensure positivity (avoiding errors)
        dist = Dirichlet(alphas)
        return dist


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.fc = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout) 

    def forward(self, inputs):
        inputs = self.drop(inputs)
        return F.log_softmax(self.bn(self.fc(inputs)), dim=1) # logarithmic probabilities align better to NLL loss (reconstruction loss)


class ProdLDA(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 hidden_size, 
                 num_topics,
                 dropout, 
                 use_lognormal=False):
        super().__init__()
        
        self.use_lognormal = use_lognormal
        self.encoder = Encoder(vocab_size, hidden_size, dropout)
        if use_lognormal:
            self.h2t = HiddenToLogNormal(hidden_size, num_topics)
        else:
            self.h2t = HiddenToDirichlet(hidden_size, num_topics)
        self.decoder = Decoder(vocab_size, num_topics, dropout)
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # For encoder layers (ReLU-like activations)
            if m == self.encoder.fc1 or m == self.encoder.fc2:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # For decoder/distribution layers
            else:
                # Smaller initialization for topic distribution layers
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, inputs):
        h = self.encoder(inputs)
        posterior = self.h2t(h)
        # Getting theta (topic distribution) from the learned posterior distribution
        if self.training:
            theta = posterior.rsample().to(inputs.device) # reparameterization trick
        else:
            theta = posterior.mean.to(inputs.device)

        if self.use_lognormal:
            theta = F.softmax(theta, dim=1)

        outputs = self.decoder(theta)
        return outputs, posterior