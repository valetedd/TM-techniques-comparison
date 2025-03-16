import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden_dim=100, dropout=0.2):
        """
        ProdLDA model - a VAE-based neural topic model
        
        Args:
            vocab_size: Size of the vocabulary
            num_topics: Number of topics
            hidden_dim: Dimension of the hidden layer
            dropout: Dropout rate
        """
        super(ProdLDA, self).__init__()
        
        # Encoder part (inference network q(θ|x))
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Mean and log variance for the variational distribution
        self.mean = nn.Linear(hidden_dim, num_topics)
        self.logvar = nn.Linear(hidden_dim, num_topics)
        
        # Decoder part (generative network p(x|θ))
        self.decoder = nn.Linear(num_topics, vocab_size)
        
        # Model parameters
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Encode the input to get mean and log variance"""
        encoded = self.encoder(x)
        mean = self.mean(encoded)
        logvar = self.logvar(encoded)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick to sample from N(mean, var) while maintaining differentiability"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        """Decode the topic proportions to reconstruct documents"""
        # Apply softmax to ensure we get a proper topic distribution
        # Note: in ProdLDA we use a softmax on the topic proportions
        z = F.softmax(z, dim=1)
        
        # Linear transformation from topic distributions to word distributions
        logits = self.decoder(z)
        
        # Apply log-softmax for numerical stability
        return F.log_softmax(logits, dim=1)
    
    def forward(self, x):
        """Forward pass through the model"""
        # Encode input
        mean, logvar = self.encode(x)
        
        # Sample topic proportions via reparameterization trick
        z = self.reparameterize(mean, logvar)
        
        # Decode to get reconstructed document
        recon = self.decode(z)
        
        return recon, mean, logvar
    
    def get_topic_word_dist(self, k=10):
        """
        Returns the top k words for each topic
        
        Args:
            k: Number of top words to return for each topic
            
        Returns:
            topic_words: List of lists containing the indices of top k words for each topic
        """
        # Get the weights from the decoder
        # These weights represent the word distributions for each topic
        topic_word_dist = self.decoder.weight.detach().T
        
        # Get the top k words for each topic
        topic_words = []
        for topic_idx in range(self.num_topics):
            _, top_word_indices = torch.topk(topic_word_dist[topic_idx], k=k)
            topic_words.append(top_word_indices.cpu().numpy().tolist())
        
        return topic_words


def train_prodlda(model, train_loader, num_epochs=100, learning_rate=0.001, device='cuda', beta=1.0):
    """
    Train the ProdLDA model
    
    Args:
        model: ProdLDA model
        train_loader: DataLoader for the training data
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        device: Device to train on ('cuda' or 'cpu')
        beta: Weight for the KL divergence term in the loss function
        
    Returns:
        model: Trained model
        losses: List of losses during training
    """
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move data to device
            data = data[0].to(device)  # Assuming data is just the document-term matrix
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mean, logvar = model(data)
            
            # Compute loss
            # Reconstruction loss (cross-entropy loss)
            recon_loss = -torch.sum(data * recon_batch, dim=1).mean()
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
            
            # Total loss
            loss = recon_loss + beta * kl_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
    
    return model, losses


def preprocess_data(docs, vocab_size=2000, min_df=5):
    """
    Preprocess documents for ProdLDA
    Simple preprocessing function - in a real application, you would want more sophisticated preprocessing
    
    Args:
        docs: List of documents (each document is a list of tokens)
        vocab_size: Maximum size of vocabulary
        min_df: Minimum document frequency for a term to be included
        
    Returns:
        doc_term_matrix: Document-term matrix (sparse)
        vocab: Dictionary mapping words to indices
    """
    from collections import Counter, defaultdict
    import scipy.sparse as sp
    
    # Count document frequency of each term
    doc_freq = defaultdict(int)
    for doc in docs:
        seen_terms = set(doc)
        for term in seen_terms:
            doc_freq[term] += 1
    
    # Filter terms by document frequency
    valid_terms = [term for term, freq in doc_freq.items() if freq >= min_df]
    
    # Sort terms by frequency (for consistent ordering)
    valid_terms = sorted(valid_terms, key=lambda x: -doc_freq[x])
    
    # Truncate vocabulary to vocab_size
    valid_terms = valid_terms[:vocab_size]
    
    # Create vocabulary
    vocab = {term: idx for idx, term in enumerate(valid_terms)}
    
    # Create document-term matrix
    rows, cols, values = [], [], []
    for doc_idx, doc in enumerate(docs):
        term_counts = Counter(doc)
        for term, count in term_counts.items():
            if term in vocab:
                rows.append(doc_idx)
                cols.append(vocab[term])
                values.append(count)
    
    doc_term_matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(docs), len(vocab)))
    
    # Normalize the document-term matrix (tf normalization)
    doc_term_matrix = doc_term_matrix.astype(float)
    row_sums = doc_term_matrix.sum(axis=1).A1
    row_indices, _ = doc_term_matrix.nonzero()
    doc_term_matrix.data /= row_sums[row_indices]
    
    return doc_term_matrix, vocab


def create_data_loader(doc_term_matrix, batch_size=64, shuffle=True):
    """
    Create a DataLoader for the document-term matrix
    
    Args:
        doc_term_matrix: Document-term matrix (scipy sparse matrix)
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        data_loader: DataLoader for the document-term matrix
    """
    # Convert sparse matrix to dense tensor
    tensor_data = torch.FloatTensor(doc_term_matrix.toarray())
    
    # Create dataset
    dataset = TensorDataset(tensor_data)
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader


def evaluate_topics(model, vocab, id2word=None, top_n=10):
    """
    Evaluate the model by printing the top words for each topic
    
    Args:
        model: Trained ProdLDA model
        vocab: Dictionary mapping words to indices
        id2word: Dictionary mapping indices to words (optional, will be created from vocab if not provided)
        top_n: Number of top words to display for each topic
    """
    # Create id2word if not provided
    if id2word is None:
        id2word = {idx: word for word, idx in vocab.items()}
    
    # Get topic-word distribution
    topic_words = model.get_topic_word_dist(k=top_n)
    
    # Print top words for each topic
    for topic_idx, word_indices in enumerate(topic_words):
        topic_str = f"Topic {topic_idx+1}: "
        topic_str += ", ".join([id2word[word_idx] for word_idx in word_indices])
        print(topic_str)


def compute_coherence(model, docs, vocab, id2word=None, measure='c_v', top_n=10):
    """
    Compute topic coherence using gensim
    
    Args:
        model: Trained ProdLDA model
        docs: List of documents (each document is a list of tokens)
        vocab: Dictionary mapping words to indices
        id2word: Dictionary mapping indices to words (optional, will be created from vocab if not provided)
        measure: Coherence measure to use ('c_v', 'u_mass', etc.)
        top_n: Number of top words to consider for coherence calculation
        
    Returns:
        coherence: Topic coherence score
    """
    try:
        import gensim
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError:
        print("gensim is required for computing coherence scores. Install with: pip install gensim")
        return None
    
    # Create id2word if not provided
    if id2word is None:
        id2word = {idx: word for word, idx in vocab.items()}
    
    # Get topic-word distribution
    topic_words = model.get_topic_word_dist(k=top_n)
    
    # Convert to list of lists of strings for gensim
    topics = [[id2word[word_idx] for word_idx in word_indices] for word_indices in topic_words]
    
    # Compute coherence
    coherence_model = CoherenceModel(topics=topics, texts=docs, dictionary=gensim.corpora.Dictionary(docs), 
                                    coherence=measure)
    coherence = coherence_model.get_coherence()
    
    return coherence


def example_usage():
    """Example usage of the ProdLDA model"""
    # Sample data (typically you would load your own data)
    import pandas as pd
    import preprocessing as pp
    docs = pp.load_pp(folder="data/UN_PP", file_or_files="tokenized.pkl")
    print(docs)
    # Parameters
    vocab_size = 1000
    num_topics = 50
    batch_size = 64
    num_epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Preprocess data
    doc_term_matrix, vocab = preprocess_data(docs, vocab_size=vocab_size)
    
    # Create data loader
    data_loader = create_data_loader(doc_term_matrix, batch_size=batch_size)
    
    # Create and train model
    model = ProdLDA(vocab_size=len(vocab), num_topics=num_topics)
    model, losses = train_prodlda(model, data_loader, num_epochs=num_epochs, device=device)
    
    # Create id2word mapping
    id2word = {idx: word for word, idx in vocab.items()}
    
    # Evaluate topics
    print("\nTop words per topic:")
    evaluate_topics(model, vocab, id2word)
    
    # Compute topic coherence
    coherence = compute_coherence(model, docs, vocab, id2word)
    print(f"\nTopic coherence: {coherence}")
    
    return model, vocab, losses


if __name__ == "__main__":
    example_usage()

# def example_usage():
#     """Example usage of the ProdLDA model"""
#     # Sample data (typically you would load your own data)
#     num_topics = 50
#     batch_size = 64
#     num_epochs = 50
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     pp_dict = pp.load_pp("data/UN_PP", ("bow.pkl", "dictionary.dict"))
#     docs, vocab = pp_dict["bow"][:500], pp_dict["index"]
#     # Create data loader
#     data_loader = create_data_loader(docs, vocab, batch_size=batch_size)
    
#     # Create and train model
#     model = ProdLDA(vocab_size=len(vocab), num_topics=num_topics)
#     model, losses = train_prodlda(model, data_loader, num_epochs=num_epochs, device=device)
    
#     # Create id2word mapping
#     id2word = vocab.id2token
#     print(id2word)
    
#     # Evaluate topics
#     print("\nTop words per topic:")
#     evaluate_topics(model, vocab, id2word)
    
#     # Compute topic coherence
#     coherence = compute_coherence(model, docs, vocab, id2word)
#     print(f"\nTopic coherence: {coherence}")
    
#     return model, vocab, losses