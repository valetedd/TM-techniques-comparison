import numpy as np
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import random
import preprocessing as pp
from nltk.corpus import stopwords
import nltk
import torch
nltk.download('stopwords')
from gensim.corpora import Dictionary
import argparse

# CLI interface
parser = argparse.ArgumentParser(description="Evaluation suite execution")
parser.add_argument('--small', action='store_true',
                    help="Work on COVID subset")
args = parser.parse_args()



class TopicEvaluationSuite:
    """
    A suite for evaluating topic models using coherence and diversity metrics.
    Compatible with LDA, prodLDA, BERTopic, and LLM-generated topics.
    """
    
    def __init__(self, texts: List[List[str]], dictionary=None, eval_top = 10):
        """
        Initialize the evaluation suite.
        
        Args:
            texts: List of tokenized documents (each document is a list of tokens)
            dictionary: Optional pre-created gensim dictionary
            eval_top: Number of top words to consider for evaluation
        """
        self.texts = texts
        if dictionary is None:
            self.dictionary = corpora.Dictionary(texts)
        else:
            self.dictionary = dictionary
        
        # Prepare corpus for coherence calculations
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        self.n_top = eval_top


    def _get_top_n_words(self, topic_model, model_type : str, n=0):
        """
        Extract top n words for each topic based on model type.
        
        Args:
            topic_model: The trained topic model
            n: Number of top words to extract
            model_type: Type of model ('lda', 'prodlda', 'bertopic', 'llm')
            
        Returns:
            List of topics, where each topic is a list of top n words;
            For LMM_TM: list of topic labels
        """
        if not n:
            n = self.n_top

        topics = []
        
        if model_type.lower() == 'lda':
            for topic_id in range(topic_model.num_topics):
                top_words = [word for word, _ in topic_model.show_topic(topic_id, n)]
                topics.append(top_words)
                
        elif model_type.lower() == 'prodlda':
            # Getting the topic matrix from the decoder of the model
            beta = topic_model.beta().numpy()
            idx2word = self.dictionary.id2token
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
            # For LLM-generated topics, assuming a list of labels
            print(f"LLM topics: {topic_model.topics}")
            return topic_model.topics
                
        return topics
    
    
    def compute_coherence(self, topic_model, model_type, coherence_measure='c_v') -> float:
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

        print(f"Computing coherence for {model_type} model")

        topics = self._get_top_n_words(topic_model, model_type=model_type)
        
        _ = self.dictionary[0]

        idx2word = self.dictionary.id2token
        texts = [[idx2word[id] for id, _ in doc] for doc in self.corpus]

        coherence_model = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=self.dictionary,
            coherence=coherence_measure
        )
        
        return coherence_model.get_coherence()
    
    def compute_topic_diversity(self, topic_model, model_type, n_words=0) -> float:
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

        if not n_words: 
            n_words = self.n_top


        topics = self._get_top_n_words(topic_model, n=n_words, model_type=model_type)
        print(model_type)
        print(topics)
        
        is_llm = model_type.lower() == 'llm'
        # Flatten the list of top words 
        all_words = [word for topic in topics for word in topic] if not is_llm else topics # llm topics are already flattened
        # Count unique words
        unique_words = set(all_words)
        
        # Diversity = unique words / total words
        diversity = len(unique_words) / len(all_words) 
        
        return diversity
    
    
    def compute_topic_overlap(self, topic_model, model_type, n_words=0) -> float:
        """
        Calculate average pairwise overlap between topics.
        
        Args:
            topic_model: The trained topic model
            model_type: The type of model
            n_words: Number of top words to consider per topic
            
        Returns:
            Average word-level overlap between topics (0-1, lower is better)
        """
        if not n_words:
            n_words = self.n_top

        
        topics = self._get_top_n_words(topic_model, n=n_words, model_type=model_type)
        
        if model_type.lower() == 'llm':
            sw = stopwords.words("english")
            topics = [[w.lower() for w in label.split(" ") if not w.lower() in sw] for label in topics]

        topic_count = len(topics)
        if topic_count <= 1:
            return 0.0
        


        # Jaccard similarity scores
        overlaps = []
        for i in range(topic_count):
            for j in range(i+1, topic_count):
                set_i = set(topics[i])
                set_j = set(topics[j])
                overlap = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                overlaps.append(overlap)
                
        return np.mean(overlaps)
    
    def autolabel_topic(self, topic_model, model_type, llm_name = 'deepseek-r1:8b', settings=None) -> List[str]:
        """
        Label topics automatically using an LLM, based on the top words.
        """
        from llm_tm import LLM_TopicModel

        if not settings:
            settings = {

                "labelling_p":"prompts/topic_labelling.txt",
                "options" : {
                    "temperature" : 0.0
                    },
                "model" : llm_name,
                }

        tm_agent = LLM_TopicModel(**settings)

        topics = self._get_top_n_words(topic_model, model_type=model_type)
        print(f"Starting labelling {model_type}: extracted these topics: {topics}")
        labels = tm_agent.label_topics(topics)
        
        return labels

    
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
        
        topics = self._get_top_n_words(topic_model, model_type=model_type)

        results = {
            'model_name': model_name,
            'model_type': model_type,
            'num_topics': len(topics),
            'topic_diversity': self.compute_topic_diversity(topic_model, model_type=model_type),
            'topic_overlap': self.compute_topic_overlap(topic_model, model_type=model_type),
            'top_topics': topics,
        }
        
        # Only compute coherence for non-LLM models
        if model_type.lower() != 'llm':
            results['coherence_cv'] = self.compute_coherence(topic_model, model_type=model_type)
            results['label'] = self.autolabel_topic(topic_model, model_type=model_type)
        else:
            results['label'] = topic_model.topics
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
        plt.show()
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

def get_data_in_timeframe(df : pd.DataFrame, timeframe : tuple[str]):
    start, end = timeframe[0], timeframe[1]
    mask = (int(start) <= df["year"]) & (df["year"] <= int(end))
    df = df[mask]
    return df["text"].to_list()



def main():
    
    # data
    df = pd.read_csv("data/UN_speeches/UNGDC_1946-2023.csv")

    if args.small:
        covid_data = get_data_in_timeframe(df, timeframe=["2020", "2022"]) # getting covid data
        random.seed(42)
        data = random.sample(covid_data, k=200)
        pp_docs = pp.basic_pp(corpus=data,
                          n_grams="tri-grams")
        dct, bow = pp.BOW_pp(docs=pp_docs, 
                         filter_extr=True, 
                         from_preprocessed=True)
    
    else:
        data = df["text"].to_list()
        dct, bow = pp.load_pp("data/UN_PP", ("bow.pkl", "dictionary.dict"))
        pp_docs = pp.load_pp("data/UN_PP", ("tokenized.pkl",))
        if not isinstance(dct, Dictionary):
            bow, dct = dct, bow
    
    # ***MODELS*** 

    NUM_TOPICS = 100

    # traditional LDA
    from LDA_baseline import train_LDA

    ITERATIONS = 1000
    PASSES = 20
    CHUNKS = 200
    classic_LDA = train_LDA(
        bow_data=bow,
        dct=dct,
        chunksize=CHUNKS,
        num_topics=NUM_TOPICS,
        iterations=ITERATIONS,
        passes=PASSES,
        eval_every=10
    )

    # prodLDA
    from prodLDA import ProdLDA, get_dataloader

    HIDDEN = 1024
    DROP_RATE = 0.2
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prodLDA = ProdLDA(
        vocab_size=len(dct),
        num_topics=NUM_TOPICS, 
        hidden=HIDDEN,
        dropout=DROP_RATE
    ).to(device)

    dataloader = get_dataloader(
        bow=bow,
        vocab_size=len(dct),
        batch_size=BATCH_SIZE,
        dct=dct
    )


    prodLDA.train_model(
        docs=dataloader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE, 
        device=device,
    )


    # Generative topic mdodelling
    # from llm_tm import LLM_TopicModel

    # AGENT = "deepseek-r1:8b"

    # options =  {"temperature" : 0.0} # no randomness

    # tm_agent = LLM_TopicModel(model=AGENT,
    #                           base_p="prompts/tm_prompt.txt",
    #                           merge_p="prompts/topic_merge.txt",
    #                           labelling_p="prompts/topic_labelling.txt",
    #                           n_topics=20,
    #                           n_shots="prompts/few_shots.txt",
    #                           seed_topics=["International Relations", "War", "Peace", "Cooperation", "Countries"],
    #                           options=options)
    # # tm_agent.get_topics(data)
    

    # bertopic 
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    import umap
    import hdbscan

    # bertopic only accepts list of strings
    bert_pp = [" ".join(doc) for doc in pp_docs]

    # Lower params for reduced size corpus
    umap_model = umap.UMAP(n_neighbors=5, metric='cosine') if args.small else None
    hdbscan_model = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=2) if args.small else None

    cv = CountVectorizer(stop_words="english", min_df=2)

    # Run BERTopic
    bertopic_model = BERTopic(
        vectorizer_model=cv,
        nr_topics=NUM_TOPICS,
        calculate_probabilities=False,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model
    )
    bertopic_model.fit_transform(bert_pp)
    

    evaluator = TopicEvaluationSuite(texts=pp_docs, eval_top=10)
    comparison_df = evaluator.compare_models({
        "LDA": (classic_LDA, "lda"),
        "prodLDA": (prodLDA, "prodlda"),
        "BERTopic": (bertopic_model, "bertopic"),
        # "LLM": (tm_agent, "llm"),
    })
    comparison_df.to_csv("data/comparison_whole.csv", index=False)
    print(comparison_df)

if __name__ == "__main__":
    main()