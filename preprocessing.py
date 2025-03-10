import spacy
model_name = "en_core_web_sm"
if not spacy.util.is_package(model_name):
    print(f"Model '{model_name}' is not installed. Please install it using:")
    print(f"python -m spacy download {model_name}")

from nltk.corpus import stopwords

import pandas as pd
import string
from pathlib import Path
import os
import pickle

import matplotlib.pyplot as plt
from typing import List, Callable

from gensim.corpora import Dictionary



def basic_pp(corpus: list[str]) -> list[list[str]]:

    tokenizer = spacy.load("en_core_web_sm")
    sw = set(stopwords.words("english"))

    processed = []
    for doc in corpus:
        doc = doc.translate(str.maketrans("", "", string.punctuation + string.digits + "\n")) # removing punctuation (X:insertions; Y:replacements; Z:deletions)
        processed.append(
            [t.text for t in tokenizer(doc)
                    if not (t.text in sw
                    or len(t.text) <= 2)]
                            )

    return processed


def LDA_pp(
        docs: List[str] | List[List[str]], 
        from_preprocessed : bool = True,
        exclusion_floor : int = 20,
        exclusion_ceil : float = 0.5):

    if not from_preprocessed:
        docs = basic_pp(docs)

    dictionary = Dictionary(docs)

    dictionary.filter_extremes(no_below=exclusion_floor, no_above=exclusion_ceil)
    bow_repr = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, bow_repr

def get_pp_data(data_path:str, func:Callable, **kwargs):
    df = pd.read_csv(filepath_or_buffer=data_path)
    texts_list = df["text"].tolist()
    pp_data = func(texts_list, **kwargs)
    return pp_data

def main():
    data = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")["text"].tolist()[:10]
    pp = basic_pp(corpus=data)
    dct, bow = LDA_pp(docs=pp)
    dct.save("data/UN_PP/dictionary.dict")

    with open("data/UN_PP/pp.pkl", mode="wb") as f:
        pickle.dump(pp, f)
    with open("data/UN_PP/bow.pkl", mode="wb") as f1:
        pickle.dump(bow, f1)


if __name__ == "__main__":
    main()
