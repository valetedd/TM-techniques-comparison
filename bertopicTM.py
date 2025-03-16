from bertopic import BERTopic
import pandas as pd
from preprocessing import basic_pp
from sklearn.feature_extraction.text import CountVectorizer

def main():
    docs = pd.read_csv("data/UN_speeches/UNGDC_1946-2023.csv")["text"].to_list()
    docs = [" ".join(doc) for doc in basic_pp(docs)]

    count_vect = CountVectorizer(stop_words="english")
    topic_model = BERTopic(
        calculate_probabilities=False,
        vectorizer_model=count_vect
        )
    topic_model.fit_transform(docs)
    result = topic_model.get_topic_info()
    print(result)
    result.drop("Representative_Docs", axis=1).to_csv("data/results/bertopics.csv")

if __name__ == "__main__":
    main()