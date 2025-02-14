from gensim.models import LdaModel
from preprocess import LDA_preprocess, basic_preprocess
import pandas as pd
from pprint import pprint as pp


def train_LDA(
        bow_data: list[str],
        dct,
        chunksize: int = 40000,
        iterations: int = 10,
        num_topics: int = 10,
        passes: int = 5,
        eval_every: int = None):
    

    print(f"Number of tokens: {len(dct)}")
    print(f"Number of docs: {len(bow_data)}")
    
    _ = dct[0]
    id2word = dct.id2token

    model = LdaModel(
        corpus=bow_data,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    return model


def LDA_coherence(model, n_topics, corpus):

    top_topics = model.top_topics(corpus)
    avg_topic_coherence = sum([t[1] for t in top_topics]) / n_topics

    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    pp(f"Top topics: {list(top_topics)}")
    return avg_topic_coherence


def main():

    df = pd.read_csv(filepath_or_buffer="UN_speeches/UNGDC_1946-2023.csv")[:100]
    texts_list = df["text"].tolist()

    processed_data = basic_preprocess(texts_list)
    dct, bow_data = LDA_preprocess(processed_data)

    # Hyperparams
    ITERATIONS = 400
    N_TOPICS = 10
    PASSES = 20

    model = train_LDA(
                bow_data=bow_data,
                dct=dct,
                iterations = ITERATIONS,
                num_topics = N_TOPICS,
                passes = PASSES,
                eval_every = None
                )
    
    coherence = LDA_coherence(model, N_TOPICS, corpus=bow_data)

if __name__ == "__main__":
    main()