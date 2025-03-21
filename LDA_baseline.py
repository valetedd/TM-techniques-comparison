from gensim.models import LdaModel
import preprocessing as pp


def train_LDA(
        bow_data: list[str],
        dct,
        chunksize: int = 20,
        iterations: int = 10,
        num_topics: int = 100,
        passes: int = 5,
        eval_every: int = None):
    
    print(f"Number of tokens: {len(dct)}")
    print(f"Number of docs: {len(bow_data)}")
    
    if not dct.id2token:
        dct.id2token = {v: k for k, v in dct.token2id.items()}

    model = LdaModel(
        corpus=bow_data,
        id2word=dct.id2token,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    return model


# def LDA_coherence(model, n_topics, corpus):

#     top_topics = model.top_topics(corpus)
#     avg_topic_coherence = sum([t[1] for t in top_topics]) / n_topics

#     return top_topics, avg_topic_coherence


def main():

    # df = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")[:500]
    # texts = df["text"].tolist()
    # dct, bow_data = BOW_pp(texts, from_preprocessed=False)

    dct, bow_data = pp.load_pp("data/UN_PP", file_or_files=("bow.pkl", "dictionary.dict"))
    bow_data = bow_data[:500]

    # Hyperparams
    ITERATIONS = 1_000
    N_TOPICS = 20
    PASSES = 20

    model = train_LDA(
                bow_data=bow_data,
                dct=dct,
                iterations = ITERATIONS,
                num_topics = N_TOPICS,
                passes = PASSES,
                )
    
    idx2w = dct.id2token
    texts = [[idx2w[i] for i, _ in tup] for tup in bow_data]
    top_topics = model.top_topics(corpus=bow_data, texts=texts, dictionary=dct, topn=10, coherence="c_v",)
    # top_topics, avg_coherence = LDA_coherence(model, N_TOPICS, corpus=bow_data)
    # print('Average topic coherence: %.4f.' % avg_coherence)
    print(f"\nTop topics:\n{"-" * 20}")
    for topic in top_topics:
        words = [w[1] for w in topic[0]]
        print(f"Words: {words}")
        print(f"Coherence: {topic[1]:.4f}\n")

    model.save("data/results/lda/lda")

if __name__ == "__main__":
    main()