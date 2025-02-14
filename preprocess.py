import nltk
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
from typing import Optional, Literal, Iterable
from tqdm.notebook import tqdm

from gensim.models import Phrases
from gensim.corpora import Dictionary

from pprint import pp




def basic_preprocess(content: list[str]) -> list[list[str]]:

    data = [word_tokenize(doc) for doc in content]

    lemmatizer = WordNetLemmatizer()
    sw = set(stopwords.words("english"))
    data = [[lemmatizer.lemmatize(t.lower()) for t in doc 
                    if not (t in sw or t.isnumeric() or len(t) <= 1)] 
                        for doc in data]

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigrams = Phrases(data)
    for idx in range(len(data)):
        for token in bigrams[data[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                data[idx].append(token)

    return data


def LDA_preprocess(docs: list[list[str]]):

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    bow_repr = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, bow_repr


def write_preproccessed(data: list[list[str]]):
    pass



def main():
    df = pd.read_csv(filepath_or_buffer="UN_speeches/UNGDC_1946-2023.csv")
    texts_list = df["text"].tolist()
    preprocessed_d =  LDA_preprocess(texts_list)

    






if __name__ == "__main__":
    main()