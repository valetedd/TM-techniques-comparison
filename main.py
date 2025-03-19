import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class TopicEvaluationSuite:
    """
    A suite for evaluating topic models using coherence and diversity metrics.
    Compatible with LDA, prodLDA, BERTopic, and LLM-generated topics.
    """
    
    def __init__(self, texts: List[List[str]], dictionary=None):
        """
        Initialize the evaluation suite.
        
        Args:
            texts: List of tokenized documents (each document is a list of tokens)
            dictionary: Optional pre-created gensim dictionary
        """
        self.texts = texts
        if dictionary is None:
            self.dictionary = corpora.Dictionary(texts)
        else:
            self.dictionary = dictionary
        
        # Prepare corpus for coherence calculations
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        
    def _get_top_n_words(self, topic_model, model_type : str, n=10):
        """
        Extract top n words for each topic based on model type.
        
        Args:
            topic_model: The trained topic model
            n: Number of top words to extract
            model_type: Type of model ('lda', 'prodlda', 'bertopic', 'llm')
            
        Returns:
            List of topics, where each topic is a list of top n words
        """
        topics = []
        
        if model_type.lower() == 'lda':
            for topic_id in range(topic_model.num_topics):
                top_words = [word for word, _ in topic_model.show_topic(topic_id, n)]
                topics.append(top_words)
                
        elif model_type.lower() == 'prodlda':
            # Getting the topic matrix from the decoder of the model
            beta = topic_model.decoder.fc.weight.cpu().detach().numpy().T
            idx2word = self.dictionary.index2word
            for i in range(len(beta)):
                topic = [idx2word[j] for j in beta[i].argsort()[:-n-1:-1]]
                topics.append(topic)
                
        elif model_type.lower() == 'bertopic':
            # For BERTopic
            for topic_id, topic in topic_model.get_topics().items():
                if topic_id != -1:  # Skip outlier topic if present
                    top_words = [word for word, _ in topic[:n]]
                    topics.append(top_words)
                    
        elif model_type.lower() == 'llm':
            # For LLM-generated topics, assuming a list of lists of words
            if isinstance(topic_model, list) and all(isinstance(topic, list) for topic in topic_model):
                topics = [topic[:n] for topic in topic_model]
            else:
                raise ValueError("LLM topics should be provided as a list of lists of words")
                
        return topics
    
    
    def compute_coherence(self, topic_model, model_type, coherence_measure='c_npmi') -> float:
        """
        Compute topic coherence for the given model.
        
        Args:
            topic_model: The trained topic model
            model_type: The type of model ('lda', 'prodlda', 'bertopic')
            coherence_measure: The coherence measure to use ('c_v', 'u_mass', 'c_npmi', 'c_uci')
            
        Returns:
            The coherence score
        """
        if model_type.lower() == 'llm':
            print("Coherence calculation not available for LLM-generated topics")
            return None
            
        topics = self._get_top_n_words(topic_model, model_type=model_type)
        
        coherence_model = CoherenceModel(
            topics=topics,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence=coherence_measure
        )
        
        return coherence_model.get_coherence()
    
    def compute_topic_diversity(self, topic_model, model_type, n_words=25) -> float:
        """
        Compute topic diversity as the percentage of unique words in the
        top N words of all topics.
        
        Args:
            topic_model: The trained topic model
            model_type: The type of model
            n_words: Number of top words to consider per topic
            
        Returns:
            Topic diversity score (0-1, higher is more diverse)
        """
        topics = self._get_top_n_words(topic_model, n=n_words, model_type=model_type)
        
        # Flatten the list of top words
        all_words = [word for topic in topics for word in topic]
        # Count unique words
        unique_words = set(all_words)
        
        # Diversity = unique words / total words
        diversity = len(unique_words) / len(all_words)
        
        return diversity
    
    
    def compute_topic_overlap(self, topic_model, model_type='lda', n_words=10) -> float:
        """
        Calculate average pairwise overlap between topics.
        
        Args:
            topic_model: The trained topic model
            model_type: The type of model
            n_words: Number of top words to consider per topic
            
        Returns:
            Average word-level overlap between topics (0-1, lower is better)
        """
        topics = self._get_top_n_words(topic_model, n=n_words, model_type=model_type)
        
        topic_count = len(topics)
        if topic_count <= 1:
            return 0.0
            
        overlaps = []
        for i in range(topic_count):
            for j in range(i+1, topic_count):
                set_i = set(topics[i])
                set_j = set(topics[j])
                overlap = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                overlaps.append(overlap)
                
        return np.mean(overlaps)
    
    def compute_pairwise_distances(self, topic_model, model_type, metric='cosine', method='embeddings', embedding_model=None):
        """
        Compute pairwise distances between topic vectors.
        
        Args:
            topic_model: The trained topic model
            model_type: The type of model
            metric: Distance metric ('cosine', 'euclidean', etc)
            method: How to represent topics ('word_dist', 'embeddings')
            embedding_model: If method='embeddings', a model to get word embeddings
            
        Returns:
            Matrix of pairwise distances
        """
        topics = self._get_top_n_words(topic_model, model_type=model_type)
        n_topics = len(topics)
        
        if method == 'word_dist':
            # Get word distributions for each topic
            word_set = set(word for topic in topics for word in topic)
            word_to_idx = {word: idx for idx, word in enumerate(word_set)}
            
            # Create sparse word vectors for each topic
            vectors = np.zeros((n_topics, len(word_set)))
            for i, topic in enumerate(topics):
                for word in topic:
                    vectors[i, word_to_idx[word]] += 1
        
        elif method == 'embeddings':
            if embedding_model is None:
                raise ValueError("Embedding model must be provided")
                
            # Get embeddings for each topic
            vectors = []
            for topic_words in topics:
                topic_embeddings = []
                for word in topic_words:
                    try:
                        topic_embeddings.append(embedding_model[word])
                    except KeyError:
                        continue  # Skip words not in the embedding model
                        
                if not topic_embeddings:
                    vectors.append(np.zeros(embedding_model.vector_size))
                else:
                    vectors.append(np.mean(topic_embeddings, axis=0))
                    
            vectors = np.array(vectors)
        
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Compute pairwise similarity matrix
        if metric == 'cosine':
            similarity_matrix = cosine_similarity(vectors)
            distance_matrix = 1 - similarity_matrix
        else:
            from sklearn.metrics import pairwise_distances
            distance_matrix = pairwise_distances(vectors, metric=metric)
            
        return distance_matrix
    
    def evaluate_model(self, topic_model, model_type, model_name=None) -> Dict:
        """
        Run a comprehensive evaluation of a topic model.
        
        Args:
            topic_model: The trained topic model
            model_type: The type of model ('lda', 'prodlda', 'bertopic', 'llm')
            model_name: Optional name for the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name is None:
            model_name = model_type
            
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'num_topics': len(self._get_top_n_words(topic_model, model_type=model_type)),
            'topic_diversity': self.compute_topic_diversity(topic_model, model_type=model_type),
            'topic_overlap': self.compute_topic_overlap(topic_model, model_type=model_type)
        }
        
        # Only compute coherence for non-LLM models
        if model_type.lower() != 'llm':
            results['coherence_cv'] = self.compute_coherence(
                topic_model, model_type=model_type, coherence_measure='c_npmi'
            )
            results['coherence_umass'] = self.compute_coherence(
                topic_model, model_type=model_type, coherence_measure='u_mass'
            )
            
        return results
    
    def compare_models(self, models_dict: Dict[str, Tuple]) -> pd.DataFrame:
        """
        Compare multiple topic models.
        
        Args:
            models_dict: Dictionary mapping model names to tuples of (model, model_type)
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        for model_name, (model, model_type) in models_dict.items():
            result = self.evaluate_model(model, model_type, model_name)
            results.append(result)
            
        return pd.DataFrame(results)
    
    def visualize_comparison(self, comparison_df: pd.DataFrame, metrics=None, figsize=(12, 6)):
        """
        Visualize model comparison results.
        
        Args:
            comparison_df: DataFrame with comparison results
            metrics: List of metrics to visualize (default: all numeric metrics)
            figsize: Figure size
        """
        if metrics is None:
            # Get all numeric columns except 'num_topics'
            metrics = [col for col in comparison_df.columns if 
                      comparison_df[col].dtype in ('int64', 'float64') and 
                      col != 'num_topics']
        
        # Prepare data for plotting
        plot_data = comparison_df.melt(
            id_vars=['model_name'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Value'
        )
        
        # Create plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='model_name', y='Value', hue='Metric', data=plot_data)
        plt.title('Topic Model Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        plt.tight_layout()
        
        return ax
    
    def top_words_table(self, models_dict: Dict[str, Tuple], n_words=10) -> pd.DataFrame:
        """
        Create a table showing top words for each topic from each model.
        
        Args:
            models_dict: Dictionary mapping model names to tuples of (model, model_type)
            n_words: Number of top words to show
            
        Returns:
            DataFrame with top words for each topic and model
        """
        all_topics = {}
        
        for model_name, (model, model_type) in models_dict.items():
            topics = self._get_top_n_words(model, n=n_words, model_type=model_type)
            all_topics[model_name] = topics
            
        # Find the maximum number of topics across models
        max_topics = max(len(topics) for topics in all_topics.values())
        
        # Create a DataFrame to hold topic words
        rows = []
        for topic_idx in range(max_topics):
            row = {'Topic #': topic_idx + 1}
            
            for model_name, topics in all_topics.items():
                if topic_idx < len(topics):
                    row[model_name] = ', '.join(topics[topic_idx])
                else:
                    row[model_name] = '-'
                    
            rows.append(row)
            
        return pd.DataFrame(rows)
    

def main():
    # data
    texts = pd.read_csv("data/UN_speeches/UNGDC_1946-2023.csv", nrows=500)["text"].to_list()

    # models 


    evaluator = TopicEvaluationSuite(texts=texts)
    # evaluator.